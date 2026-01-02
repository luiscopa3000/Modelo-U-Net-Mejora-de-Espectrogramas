import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader



def lista_to_dataloader(lista):
    dataloader = DataLoader(
        lista,
        batch_size=32,
        shuffle=False
    )
    return dataloader


def normalizar_espectrograma_mel(espectrograma):
    """
    Normaliza un espectrograma mel para que sus valores estén en el rango [0, 1].
    
    Args:
    espectrograma (torch.Tensor): El espectrograma mel (tensor 2D o 3D).
    
    Returns:
    torch.Tensor: Espectrograma normalizado en el rango [0, 1].
    """
    # Encontrar el valor mínimo y máximo del espectrograma
    min_val = espectrograma.min()
    max_val = espectrograma.max()
    
    # Normalizar el espectrograma al rango [0, 1]
    espectrograma_normalizado = (espectrograma - min_val) / ((max_val - min_val)+1e-8)
    
    return espectrograma_normalizado

# =============================
# Función para graficar comparaciones de espectrogramas
# =============================
def plot_comparative_spectrograms(degraded, fake, clean, n_samples=5):
    degraded = degraded.cpu()
    fake = fake.cpu()
    clean = clean.cpu()

    fig, axs = plt.subplots(3, n_samples, figsize=(15, 8))
    for i in range(n_samples):
        degraded_np = degraded[i].detach().numpy()
        fake_np = fake[i].detach().numpy()
        clean_np = clean[i].detach().numpy()

        axs[0, i].imshow(degraded_np[0], aspect='auto', origin='lower', cmap='inferno')
        axs[0, i].set_title(f'Muestra {i+1} - Degraded')

        axs[1, i].imshow(fake_np[0], aspect='auto', origin='lower', cmap='inferno')
        axs[1, i].set_title(f'Muestra {i+1} - Fake')

        axs[2, i].imshow(clean_np[0], aspect='auto', origin='lower', cmap='inferno')
        axs[2, i].set_title(f'Muestra {i+1} - Clean')

    plt.tight_layout()
    plt.show()


def plot_mel_spectrogram(mel_spectrogram_db):
    """
    Función para graficar un espectrograma Mel de forma torch.Size([1, 128, 16952]).
    
    Args:
    mel_spectrogram_db (Tensor): Tensor 3D de tamaño [1, 128, N] (N es la dimensión temporal)
    """
    # Asegurarnos de que estamos trabajando con el primer canal (en caso de que sea mono o esté en 3D)
    mel_spectrogram_db = mel_spectrogram_db[0]  # Seleccionar el primer canal

    # Configuración de la visualización
    plt.figure(figsize=(12, 6))
    plt.imshow(mel_spectrogram_db.numpy(), aspect='auto', origin='lower', cmap='inferno')

    # Etiquetas y título
    plt.title("Espectrograma Mel")
    plt.xlabel("Tiempo")
    plt.ylabel("Frecuencia Mel")
    
    # Añadir barra de color
    plt.colorbar(format="%+2.0f dB")
    
    # Mostrar gráfico
    plt.tight_layout()
    plt.savefig('prueba.jpg')
    plt.show()


def plot_two_mel_spectrograms(mel1, mel2):
    """
    Grafica dos espectrogramas Mel en la misma figura, uno al lado del otro.

    Args:
        mel1 (Tensor): Primer espectrograma Mel, tamaño [1, 128, N]
        mel2 (Tensor): Segundo espectrograma Mel, tamaño [1, 128, M]
    """
    # Seleccionar el primer canal en ambos casos
    mel1 = mel1[0]
    mel2 = mel2[0]

    # Crear figura con dos subplots
    fig, axes = plt.subplots(2, 1, figsize=(15, 7))

    # Primer espectrograma
    im1 = axes[0].imshow(mel1.numpy(), aspect='auto', origin='lower', cmap='inferno')
    axes[0].set_title("Espectrograma Mel 1")
    axes[0].set_xlabel("Tiempo")
    axes[0].set_ylabel("Frecuencia Mel")
    fig.colorbar(im1, ax=axes[0], format="%+2.0f dB")

    # Segundo espectrograma
    im2 = axes[1].imshow(mel2.numpy(), aspect='auto', origin='lower', cmap='inferno')
    axes[1].set_title("Espectrograma Mel 2")
    axes[1].set_xlabel("Tiempo")
    axes[1].set_ylabel("Frecuencia Mel")
    fig.colorbar(im2, ax=axes[1], format="%+2.0f dB")

    # Ajustar diseño
    plt.tight_layout()
    plt.show()








