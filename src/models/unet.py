import os
import torch
import torch.nn as nn
import torch.nn.functional as F


class FrequencyAwareConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Convolución vertical (frecuencias)
        self.conv = nn.Conv2d(in_channels, out_channels, (5,1), padding=(2,0))
        # Convolución horizontal (tiempo)
        self.convh = nn.Conv2d(out_channels, out_channels, (1,3), padding=(0,1))
        self.norm = nn.InstanceNorm2d(out_channels)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)  # Captura relaciones verticales (armónicos)
        x = self.convh(x)  # Captura contexto temporal
        return self.activation(self.norm(x))

class ASPP(nn.Module):
    def __init__(self, in_channels=256, out_channels=256, rates=[1, 2, 4], dropout_rate=0.2):
        super().__init__()
        self.dropout_rate = dropout_rate
        # 1. Depthwise Dilatadas
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 3, padding=r, dilation=r, groups=in_channels, bias=False),
                nn.Conv2d(in_channels, out_channels, 1),
                nn.GroupNorm(8, out_channels),
                nn.ReLU()
            ) for r in rates
        ])
        # 2. Max-Pooling Multi-Escala
        self.max_pools = nn.ModuleList([
            nn.Sequential(
                nn.MaxPool2d(kernel_size=k, stride=1, padding=k//2),
                nn.Conv2d(in_channels, out_channels, 1),
                nn.ReLU()
            ) for k in [3, 5]
        ])
        # 3. Fusión + Dropout
        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels * (len(rates) + len(self.max_pools)), out_channels, 1),
            nn.Dropout2d(self.dropout_rate)
        )

    def forward(self, x):
        features = [conv(x) for conv in self.convs]
        features += [pool(x) for pool in self.max_pools]
        out = self.fusion(torch.cat(features, dim=1))
        return out

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.0):
        super().__init__()
        groups = max(1, out_channels // 8)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(groups, out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(groups, out_channels)
        )
        self.shortcut = (nn.Conv2d(in_channels, out_channels, 1)
                         if in_channels != out_channels else nn.Identity())
        self.act = nn.ReLU()
        self.dropout = nn.Dropout2d(dropout_rate)

    def forward(self, x):
        out = self.conv(x) + self.shortcut(x)
        out = self.act(out)
        out = self.dropout(out)
        return out


class DoubleConv(nn.Module):
    """
    Applies two consecutive convolutional layers each followed by ReLU activation.
    """
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class Down(nn.Module):
    """
    A downscaling block that returns both the convolution output (for skip connection)
    and the pooled output.
    """
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.double_conv = DoubleConv(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        conv_out = self.double_conv(x)
        pooled = self.pool(conv_out)
        return conv_out, pooled


class Up(nn.Module):
    """
    An upscaling block with transposed convolution followed by a double conv.
    """
    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()
        # Transposed conv for upsampling
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        # After concatenation, channels = out_channels * 2
        self.double_conv = DoubleConv(in_channels, out_channels)
        

    def forward(self, x, skip_connection):
        x = self.up(x)
        # Input patches may differ by one pixel due to rounding, pad if necessary
        diffY = skip_connection.size(2) - x.size(2)
        diffX = skip_connection.size(3) - x.size(3)
        if diffY != 0 or diffX != 0:
            x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                          diffY // 2, diffY - diffY // 2])
        # Concatenate along channel dimension
        x = torch.cat([skip_connection, x], dim=1)
        return self.double_conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_chanel=32):
        super(UNet, self).__init__()
        # Encoder path
        self.down1 = Down(in_channels, base_chanel)
        self.down2 = Down(base_chanel, base_chanel * 2)
        self.down3 = Down(base_chanel * 2, base_chanel * 4)
        self.down4 = Down(base_chanel * 4, base_chanel * 8)

        # Bottleneck
        #self.bottleneck = DoubleConv(base_chanel * 8, base_chanel * 16)
        self.bottleneck = nn.Sequential(
            ASPP(base_chanel*8, base_chanel*8, rates=[1,2,4]),
            ResidualBlock(base_chanel*8, base_chanel*16),
            FrequencyAwareConv(base_chanel *16, base_chanel*16)
            
        )

        # Decoder path
        self.up1 = Up(base_chanel * 16, base_chanel * 8)
        self.up2 = Up(base_chanel * 8, base_chanel * 4)
        self.up3 = Up(base_chanel * 4, base_chanel * 2)
        self.up4 = Up(base_chanel * 2, base_chanel)

        # Final output convolution
        self.final_conv = nn.Conv2d(base_chanel, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        c1, p1 = self.down1(x)  # conv and pool
        c2, p2 = self.down2(p1)
        c3, p3 = self.down3(p2)
        c4, p4 = self.down4(p3)

        # Bottleneck
        bn = self.bottleneck(p4)

        # Decoder with skip connections
        u1 = self.up1(bn, c4)
        u2 = self.up2(u1, c3)
        u3 = self.up3(u2, c2)
        u4 = self.up4(u3, c1)

        # Final 1x1 conv
        output = self.final_conv(u4)
        return torch.sigmoid(output)




class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_chanel=32):
        super(UNet, self).__init__()
        # Encoder path
        self.down1 = Down(in_channels, base_chanel)
        self.down2 = Down(base_chanel, base_chanel * 2)
        self.down3 = Down(base_chanel * 2, base_chanel * 4)
        self.down4 = Down(base_chanel * 4, base_chanel * 8)

        # Bottleneck
        #self.bottleneck = DoubleConv(base_chanel * 8, base_chanel * 16)
        self.bottleneck = nn.Sequential(
            ASPP(base_chanel*8, base_chanel*8, rates=[1,2,4]),
            ResidualBlock(base_chanel*8, base_chanel*16),
            FrequencyAwareConv(base_chanel *16, base_chanel*16)
            
        )

        # Decoder path
        self.up1 = Up(base_chanel * 16, base_chanel * 8)
        self.up2 = Up(base_chanel * 8, base_chanel * 4)
        self.up3 = Up(base_chanel * 4, base_chanel * 2)
        self.up4 = Up(base_chanel * 2, base_chanel)

        # Final output convolution
        self.final_conv = nn.Conv2d(base_chanel, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        c1, p1 = self.down1(x)  # conv and pool
        c2, p2 = self.down2(p1)
        c3, p3 = self.down3(p2)
        c4, p4 = self.down4(p3)

        # Bottleneck
        bn = self.bottleneck(p4)

        # Decoder with skip connections
        u1 = self.up1(bn, c4)
        u2 = self.up2(u1, c3)
        u3 = self.up3(u2, c2)
        u4 = self.up4(u3, c1)

        # Final 1x1 conv
        output = self.final_conv(u4)
        return torch.sigmoid(output)


# Función para cargar el modelo entrenado
def cargar_modelo(path_modelo: str, device: torch.device = None):
    modelo = UNet().to(device)
    checkpoint = torch.load(path_modelo, map_location=device)
    modelo.load_state_dict(checkpoint['generator_state_dict'])
    modelo.eval()  # Coloca el modelo en modo evaluación
    return modelo







