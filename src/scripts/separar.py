import torch
import torchaudio
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.unet import UNet
from src.models.disc import Discriminator
from src.models.demucsm import SepararAudio, separar_demucs, cargar_demucs, separar_eval_demucs


from src.data.preprocesar import mel_spectrogram
from src.data.preprocesar import dividir_en_segmentos
from src.utils.datos import plot_mel_spectrogram
from src.utils.datos import plot_comparative_spectrograms

def normalizar_lista_espectrogramas(lista_espectrogramas):
    """
    Normaliza una lista de espectrogramas Mel con la fórmula:
    (input_tensor - input_tensor.min()) / (input_tensor.max() - input_tensor.min())
    Y guarda los valores min y max originales con cada espectrograma.
    
    Args:
        lista_espectrogramas: Lista de tensores, donde cada tensor tiene la forma [1, 128, 128]
        
    Returns:
        lista_normalizada: Lista de tuplas (espectrograma_normalizado, min_original, max_original).
    """
    lista_normalizada = []
    
    for espectrograma in lista_espectrogramas:
        # Asegúrate de que el tensor sea un float para evitar problemas con la normalización
        espectrograma = espectrograma.float()

        # Obtener los valores min y max originales
        min_val = espectrograma.min()
        max_val = espectrograma.max()

        # Normalización: (tensor - min) / (max - min)
        espectrograma_normalizado = (espectrograma - min_val) / (max_val - min_val)
        
        # Añadir a la lista como una tupla (espectrograma_normalizado, min_val, max_val)
        lista_normalizada.append((espectrograma_normalizado, min_val, max_val))
    
    return lista_normalizada


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
    espectrograma_normalizado = (espectrograma - min_val) / (max_val - min_val)
    
    return espectrograma_normalizado

def desnormalizar_espectrograma(espectrograma_normalizado, min_val, max_val):
    # Desnormalizamos usando la fórmula inversa
    espectrograma_desnormalizado = (espectrograma_normalizado * (max_val - min_val)) + min_val
    return espectrograma_desnormalizado

# Función para cargar el modelo entrenado
def cargar_modelo(path_modelo: str, device: torch.device = None):
    modelo = UNet().to(device)
    checkpoint = torch.load(path_modelo, map_location=device)
    modelo.load_state_dict(checkpoint['generator_state_dict'])
    modelo.eval()  # Coloca el modelo en modo evaluación
    return modelo


def cargar_vocoder():
    from src.models.big_v_gan import bigvgan

    model = bigvgan.BigVGAN.from_pretrained('nvidia/bigvgan_v2_44khz_128band_512x', use_cuda_kernel=False)
    model.remove_weight_norm()
    model = model.eval().to(device)
    return model

def procesar_unet(data, model):
    min_val = data.min()
    max_val = data.max()
    data2 = dividir_en_segmentos(normalizar_espectrograma_mel(data))
    ls_temp = []
    with torch.inference_mode():
        for mel in tqdm(data2, leave=False):
            mel = mel.to(device='cuda')
            wav_gen = model(mel.unsqueeze(0))
            ls_temp.append(wav_gen.squeeze(0))

    mel_t = torch.concat(ls_temp, dim=2)
    mel_t = mel_t * (max_val - min_val) + min_val
    return mel_t


def procesar_vocoder(data, model):
    data = split_spectrogram(data)
    ls_temp = []
    with torch.inference_mode():
        i = 0
        for mel in tqdm(data, leave=False):
            mel = mel.to(device='cuda')
            wav_gen = model(mel)
            ls_temp.extend(list(wav_gen))
            if i == 10:
                break
            i+=1
    salida = unir_wav(ls_temp)
    return salida

def split_spectrogram(spec: torch.Tensor, window_size: int = 128, overlap_frames: int = 20):
    """
    Divide un espectrograma de forma [1,128,n_frames] en una lista de espectrogramas [1,128,window_size],
    con un solapamiento fijo en número de frames.

    Parámetros:
        spec (torch.Tensor): Tensor con forma [1,128,n_frames]
        window_size (int): Tamaño de ventana (default 128)
        overlap_frames (int): Número de frames que se solapan entre ventanas consecutivas

    Retorna:
        List[torch.Tensor]: Lista de tensores con forma [1,128,window_size]
    """
    assert 0 <= overlap_frames < window_size, "El solapamiento debe ser menor que el tamaño de ventana"

    stride = window_size - overlap_frames
    n_frames = spec.shape[2]

    spec_windows = []
    for start in range(0, n_frames - window_size + 1, stride):
        end = start + window_size
        window = spec[:, :, start:end]
        spec_windows.append(window)

    return spec_windows


def crossfade_waveforms(w1: torch.Tensor, w2: torch.Tensor, overlap: int) -> torch.Tensor:
    """
    Une dos waveforms con una zona de solapamiento usando crossfade con ventana Hann.

    Parámetros:
        w1 (torch.Tensor): Primer waveform, forma [1, N]
        w2 (torch.Tensor): Segundo waveform, forma [1, M]
        overlap (int): Tamaño de la zona solapada

    Retorna:
        torch.Tensor: Waveform unido con crossfade
    """
    assert w1.shape[0] == 1 and w2.shape[0] == 1, "Se espera forma [1, N]"
    assert w1.shape[1] >= overlap and w2.shape[1] >= overlap, "Los waveforms deben tener suficiente solapamiento"

    # Zonas relevantes
    non_overlap_1 = w1[:, :-overlap]
    overlap_1 = w1[:, -overlap:]

    overlap_2 = w2[:, :overlap]
    non_overlap_2 = w2[:, overlap:]

    # Ventanas de crossfade
    fade_out = torch.linspace(1, 0, steps=overlap).to(w1.device)
    fade_in = torch.linspace(0, 1, steps=overlap).to(w1.device)

    # Aplicar ventanas
    crossfaded = (overlap_1 * fade_out) + (overlap_2 * fade_in)
    crossfaded = crossfaded.unsqueeze(0) if crossfaded.ndim == 1 else crossfaded

    # Unir todo
    joined = torch.cat([non_overlap_1, crossfaded, non_overlap_2], dim=1)
    return joined

def unir_wav(ls_wavs):
    wav = ls_wavs[0]
    for i in range(1,len(ls_wavs)):
        wav = crossfade_waveforms(wav, ls_wavs[i], 10240)
    return wav


def procesar(demucs, unet_d, unet_b, vocoder, path_audio):
    out = separar_demucs(path_audio, demucs)
    drums = out[0]
    bass = out[1]
    others = out[2]
    vocals = out[3]

    #Espectrogramas mel
    drums_mel = mel_spectrogram(drums)
    bass_mel = mel_spectrogram(bass)
    others_mel = mel_spectrogram(others)
    vocals_mel = mel_spectrogram(vocals)
    
    data = procesar_unet(drums_mel, unet_d)
    drums = procesar_vocoder(data, vocoder)
    
    data = procesar_unet(bass_mel, unet_b)
    bass = procesar_vocoder(data, vocoder)

    return [drums, bass, others, vocals]


def evaluar(demucs, unet_d, unet_b, wav):
    out = separar_eval_demucs(wav, demucs)
    for i in range(len(out)):
        if out[i].shape[0] != 1:
            out[i] = torch.mean(out[i], dim=0, keepdim=True)

    drums = out[0]
    bass = out[1]
    others = out[2]
    vocals = out[3]



    #Espectrogramas mel
    drums_mel = mel_spectrogram(drums)
    bass_mel = mel_spectrogram(bass)
    others_mel = mel_spectrogram(others)
    vocals_mel = mel_spectrogram(vocals)
    
    others_mel = dividir_en_segmentos(others_mel)
    vocals_mel = dividir_en_segmentos(vocals_mel)
    
    
    drums_mel = procesar_unet(drums_mel, unet_d)
    bass_mel = procesar_unet(bass_mel, unet_b)

    others_mel = torch.concat(others_mel, dim=2)
    vocals_mel = torch.concat(vocals_mel, dim=2)
    
    return [drums_mel, bass_mel, others_mel, vocals_mel]

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    unet_d = cargar_modelo(R"data\drums_entrenmiento\checkpoint_epoch_50.pth", device)
    unet_b = cargar_modelo(R"data\bass_entrenmiento\checkpoint_epoch_50.pth", device)
    vocoder = cargar_vocoder()
    demucs = cargar_demucs()

    path_audio = R"C:\Users\Luis\Documents\DATASETS\musdb18hq\test\target\all_01.wav"
    data = procesar(demucs, unet_d, unet_b, vocoder, path_audio)



