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
from src.models.unet import UNet
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

def train(generator, discriminator, dataloader,val_loader, num_epochs, device, checkpoint_path=None):
    # 2) Creamos optimizer con dos grupos de parámetros
    opt_g =  optim.Adam(generator.parameters(), lr=1e-4, betas=(0.5, 0.999), weight_decay=1e-2)

    #opt_g = optim.Adam(generator.parameters(), lr=2e-4, betas=(0.5, 0.999), weight_decay=1e-2)
    opt_d = optim.Adam(discriminator.parameters(), lr=1e-4, betas=(0.5, 0.999), weight_decay=1e-2)
    adversarial_loss = nn.BCELoss()
    reconstruction_loss = nn.L1Loss()

    if checkpoint_path != None:
        epoch_carg, g_loss_epoch, d_loss_epoch, avg_adv_loss, avg_l1_loss, g_val_loss_epoch, d_val_loss_epoch, val_avg_adv, val_avg_l1 = load_checkpoint(
        generator, discriminator, opt_g, opt_d, checkpoint_path, device)
    else:
        epoch_carg =0

    for epoch in range(epoch_carg+1, num_epochs+1):
        g_loss_epoch = 0.0
        d_loss_epoch = 0.0
        avg_l1_loss = 0.0
        avg_adv_loss = 0.0

        generator.train()
        discriminator.train()

        ultimo = None
        for real, target in tqdm(dataloader, desc=f'Epoch [{epoch}/{num_epochs}]', leave=False):
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


        g_loss_epoch /= len(dataloader)
        d_loss_epoch /= len(dataloader)
        avg_adv_loss /= len(dataloader)
        avg_l1_loss /= len(dataloader)

        print(f"Epoch [{epoch}/{num_epochs}]  G_loss: {g_loss_epoch:.4f}  D_loss: {d_loss_epoch:.4f} ")
 

        generator.eval()
        discriminator.eval()
        #out = generator(real)
        #plot_comparative_spectrograms(real, out, target)

        g_val_loss_epoch = 0.0  # Pérdida total del generador en validación
        d_val_loss_epoch = 0.0  # Pérdida total del discriminador en validación
        val_avg_adv = 0.0
        val_avg_l1 = 0.0

        with torch.no_grad():  # Desactiva el cálculo de gradientes durante la validación
            for real, target in tqdm(val_loader, desc=f'Validation Epoch [{epoch}/{num_epochs}]', leave=False):
                real = real.to(device)
                target = target.to(device)
                batch_size = real.size(0)

                # Labels
                valid = torch.ones(batch_size, 1, 7, 7, device=device)  # Etiquetas reales
                fake_lbl = torch.zeros(batch_size, 1, 7, 7, device=device)  # Etiquetas falsas

                # ------------------
                # Validación del Generador
                # ------------------
                fake = generator(real)
                pred_fake = discriminator(fake)
                g_adv_val = adversarial_loss(pred_fake, valid)  # Pérdida adversarial del generador
                g_rec_val = reconstruction_loss(fake, target)  # Pérdida de reconstrucción del generador
                g_val_loss = g_adv_val + 100 * g_rec_val  # Pérdida total del generador en validación

                # ----------------------
                # Validación del Discriminador
                # ----------------------
                pred_real = discriminator(target)
                pred_fake_det = discriminator(fake.detach())
                d_real_val = adversarial_loss(pred_real, valid)  # Pérdida adversarial del discriminador con imágenes reales
                d_fake_val = adversarial_loss(pred_fake_det, fake_lbl)  # Pérdida adversarial del discriminador con imágenes falsas
                d_val_loss = 0.5 * (d_real_val + d_fake_val)  # Pérdida total del discriminador en validación

                # Acumulamos las pérdidas
                g_val_loss_epoch += g_val_loss.item()
                d_val_loss_epoch += d_val_loss.item()
                val_avg_adv += g_adv_val.item()
                val_avg_l1 += g_rec_val.item()
            #plot_comparative_spectrograms(real, fake, target)

            # Promediamos las pérdidas de la validación
            g_val_loss_epoch /= len(val_loader)
            d_val_loss_epoch /= len(val_loader)
            val_avg_adv /= len(val_loader)
            val_avg_l1 /= len(val_loader)

            # Imprimimos las pérdidas promedio de validación
            print(f"Epoch [{epoch}/{num_epochs}]  G_val_loss: {g_val_loss_epoch:.4f}  D_val_loss: {d_val_loss_epoch:.4f}")




            # Guardar el checkpoint después de cada época
            checkpoint_path = f'data/entrenmiento/checkpoint_epoch_{epoch}.pth'
            save_checkpoint(generator, discriminator, opt_g, opt_d, epoch, g_loss_epoch, d_loss_epoch, 
                            avg_adv_loss, avg_l1_loss, g_val_loss_epoch, d_val_loss_epoch, 
                            val_avg_adv, val_avg_l1, checkpoint_path)

            # Después de calcular las métricas de la época, llamamos a la función para guardarlas
            save_metrics_to_csv(
                epoch, g_loss_epoch, d_loss_epoch, avg_adv_loss, avg_l1_loss,
                g_val_loss_epoch, d_val_loss_epoch, val_avg_adv, val_avg_l1
            )



#############################################
# MAIN USO Y VERIFICACIÓN DE DIMENSIONES
#############################################

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Inicia...")
    dataset = Audio_dataset(csv_file=R"data\precomputed\train.csv")

    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    data_val = Audio_dataset(csv_file=R"data\precomputed\valid.csv")

    val_loader = DataLoader(
        data_val,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )


    model = UNet().to(device)
    discriminador = Discriminator().to(device)
    
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total de parámetros: {total_params/1e6:.2f}M")
    train(model, discriminador, dataloader, val_loader,100, device)



