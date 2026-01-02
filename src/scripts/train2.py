import os
import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM
from torch.utils.data import DataLoader




from src.data.dataset import Audio_dataset
from src.utils.data_augment import quitar_frecuencias 
from src.utils.modelo import load_checkpoint, save_checkpoint, save_metrics_to_csv
from src.utils.datos import plot_comparative_spectrograms
from src.models.unetv2 import UNet
from src.models.disc import Discriminator


import random
import torch.nn.functional as F
def recortar_e_interpolar(tensor1, tensor2, max_recorte=60):
    """
    Recorta un número aleatorio de columnas del final (entre 0 y max_recorte) de dos tensores alineados
    y luego interpola para restaurar su tamaño original.

    Parámetros:
    - tensor1, tensor2: Tensores de forma [B, 1, F, T]
    - max_recorte: Máximo número de columnas que pueden recortarse (aleatoriamente)

    Devuelve:
    - tensor1_interp, tensor2_interp: Tensores con la forma original [B, 1, F, T]
    """
    assert tensor1.shape == tensor2.shape, "Los tensores deben tener la misma forma"
    B, C, freq_bins, time_bins = tensor1.shape

    recorte = random.randint(0, max_recorte-1)
    time_recortado = time_bins - recorte
    assert time_recortado > 0, f"Recorte aleatorio demasiado grande para T={time_bins}"

    # Recorte
    tensor1_cortado = tensor1[:, :, :, :time_recortado]
    tensor2_cortado = tensor2[:, :, :, :time_recortado]

    # Interpolación para restaurar forma original
    tensor1_interp = F.interpolate(tensor1_cortado, size=(freq_bins, time_bins), mode='bilinear', align_corners=False)
    tensor2_interp = F.interpolate(tensor2_cortado, size=(freq_bins, time_bins), mode='bilinear', align_corners=False)

    return tensor1_interp, tensor2_interp

# ---------------------
#   Training Function
# ---------------------

def train(generator, discriminator, dataloader, num_epochs, device):
    # 2) Creamos optimizer con dos grupos de parámetros
    opt_g =  optim.Adam(generator.parameters(), lr=1e-4, betas=(0.5, 0.999), weight_decay=1e-5)

    #opt_g = optim.Adam(generator.parameters(), lr=2e-4, betas=(0.5, 0.999), weight_decay=1e-2)
    opt_d = optim.Adam(discriminator.parameters(), lr=1e-4, betas=(0.5, 0.999), weight_decay=1e-5)
    adversarial_loss = nn.BCELoss()
    reconstruction_loss = nn.L1Loss()

    for epoch in range(1, num_epochs+1):
        g_loss_epoch = 0.0
        d_loss_epoch = 0.0
        avg_l1_loss = 0.0
        avg_adv_loss = 0.0

        generator.train()
        discriminator.train()

        ultimo = None
        for idx, (real, target) in enumerate(tqdm(dataloader, desc=f'Epoch [{epoch}/{num_epochs}]', leave=False)):
            real = real.to(device)
            target = target.to(device)
            batch_size = real.size(0)

            #real = quitar_frecuencias(real, min_ancho=3, max_ancho=30, cantidad_lineas=3, min_fila=20)
            #real, target = recortar_e_interpolar(real, target, max_recorte=30)

            # Labels
            valid = torch.ones(batch_size, 1,7,7, requires_grad=False, device=device)
            fake_lbl = torch.zeros(batch_size, 1,7,7, requires_grad=False, device=device)

            # ------------------
            # Train Generator
            # ------------------
            opt_g.zero_grad()
            fake = generator(real)
            pred_fake = discriminator(fake)
            g_adv = adversarial_loss(pred_fake, valid)
            g_rec = reconstruction_loss(fake, target)
            g_loss = g_adv + 100 * g_rec
            g_loss.backward()
            opt_g.step()

            # ----------------------
            # Train Discriminator
            # ----------------------
            
            opt_d.zero_grad()
            pred_real = discriminator(target)
            pred_fake_det = discriminator(fake.detach())
            d_real = adversarial_loss(pred_real, valid)
            d_fake = adversarial_loss(pred_fake_det, fake_lbl)
            d_loss = 0.5 * (d_real + d_fake)
            d_loss.backward()
            opt_d.step()

            d_loss_epoch += d_loss.item()
            g_loss_epoch += g_loss.item()
            avg_adv_loss += g_adv.item()
            avg_l1_loss += g_rec.item()
            if idx == len(dataloader)-2:
                plot_comparative_spectrograms(real, fake, target)


        g_loss_epoch /= len(dataloader)
        d_loss_epoch /= len(dataloader)
        avg_adv_loss /= len(dataloader)
        avg_l1_loss /= len(dataloader)

        print(f"Epoch [{epoch}/{num_epochs}]  G_loss: {g_loss_epoch:.4f}  D_loss: {d_loss_epoch:.4f} ")



#############################################
# MAIN USO Y VERIFICACIÓN DE DIMENSIONES
#############################################

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Inicia...")
    dataset = Audio_dataset(csv_file=R"data\precomputed\prueba.csv")

    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=os.cpu_count(),
        pin_memory=True,
        persistent_workers=True
    )
    model = UNet().to(device)
    discriminador = Discriminator().to(device)
    
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total de parámetros: {total_params/1e6:.2f}M")
    train(model, discriminador, dataloader,100, device)


