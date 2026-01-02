import torch
import torchaudio
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.unet import UNet
from src.models.disc import Discriminator
from src.models.demucsm import SepararAudio, separar_demucs


from src.data.preprocesar import mel_spectrogram
from src.data.preprocesar import dividir_en_segmentos
from src.utils.datos import plot_mel_spectrogram
from src.utils.datos import plot_comparative_spectrograms


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def lista_to_dataloader(lista):
    dataloader = DataLoader(
        lista,
        batch_size=32,
        shuffle=False,
        num_workers=4
    )
    return dataloader

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
    ls_temp = []
    with torch.inference_mode():
        for mel in tqdm(data):
            mel = mel.to(device='cuda')
            wav_gen = model(mel)
            ls_temp.extend(list(wav_gen))

    return ls_temp


import torch.nn.functional as F
def procesar_vocoder(data, model):
    ls_temp = []
    with torch.inference_mode():
        i = 0
        for mel in tqdm(data):
            mel = mel.to(device='cuda')
            wav_gen = model(mel)
            ls_temp.extend(list(wav_gen))
            if i == 10:
                break
            i+=1

    return ls_temp




def trocear_tensor(tensor):
    # Verifica las dimensiones del tensor
    _, _, n_frames = tensor.shape
    
    # Lista para guardar los espectrogramas troceados
    espectrogramas = []
    
    # Trocear el tensor en fragmentos de tamaño [1, 128, 80]
    for i in range(0, n_frames, 80):
        fragmento = tensor[:, :, i:i+80]  # Seccionamos de 80 en 80
        espectrogramas.append(fragmento)
    
    return espectrogramas



# Load model directly
from transformers import SpeechT5HifiGan

def separar(path=None):
    #hifi_gan = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
    #hifi_gan = hifi_gan.to(device)
    if path != None:
        unet = cargar_modelo(R'data\bass_entrenmiento\checkpoint_epoch_50.pth', device)
        vocoder = cargar_vocoder()
        out = separar_demucs(path)
        drums = out[0]
        bass = out[1]
        others = out[2]
        vocals = out[3]

        #Espectrogramas mel
        drums_mel = mel_spectrogram(drums)
        bass_mel = mel_spectrogram(bass)
        others_mel = mel_spectrogram(others)
        vocals_mel = mel_spectrogram(vocals)
        
        #Esctraer minimos y maximos
        min_val = drums_mel.min()
        max_val = drums_mel.max()

        #Espectrogramas mel por chunks
        drums_mel_ls= dividir_en_segmentos(normalizar_espectrograma_mel(drums_mel))
        bass_mel_ls = dividir_en_segmentos(normalizar_espectrograma_mel(bass_mel))
        others_mel_ls = dividir_en_segmentos(normalizar_espectrograma_mel(others_mel))
        vocals_mel_ls = dividir_en_segmentos(normalizar_espectrograma_mel(vocals_mel))

        data = lista_to_dataloader(bass_mel_ls)
        data = procesar_unet(data, unet)

        mel_t = torch.concat(data, dim=2)
        # Desnormalizar el espectrograma
        mel_t = mel_t * (max_val - min_val) + min_val
        #lista_mel = dividir_en_segmentos(mel_t)
        lista_mel = trocear_tensor(mel_t)

        salida = procesar_vocoder(lista_mel, vocoder)
        for data in salida:
            print(data.shape)

        salida = torch.concat(salida, dim=1)
        #print(salida[0].shape)


        wav_gen_float = salida.cpu() # wav_gen es un FloatTensor con forma [1, T_time]

        
        # Guardar el tensor float directamente (sin convertir a int16)
        torchaudio.save(
            "bass1.wav",
            wav_gen_float,  # Tensor float32 con forma [1, T] (valores en [-1, 1])
            sample_rate=44100,  # 44000 Hz
            encoding="PCM_S",  # Formato estándar para WAV
            bits_per_sample=16,  # 16 bits (calidad estándar)
        )



import re
import os



def medir(path=None):
    #hifi_gan = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
    #hifi_gan = hifi_gan.to(device)
    if path != None:
        unet = cargar_modelo(R'data\entrenmiento\checkpoint_epoch_50.pth', device)

        out = separar_demucs(path)
        drums = out[0]
        bass = out[1]
        others = out[2]
        vocals = out[3]

        #Espectrogramas mel
        drums_mel = mel_spectrogram(drums)
        bass_mel = mel_spectrogram(bass)
        others_mel = mel_spectrogram(others)
        vocals_mel = mel_spectrogram(vocals)
        
        #Esctraer minimos y maximos
        min_val = drums_mel.min()
        max_val = drums_mel.max()

        #Espectrogramas mel por chunks
        drums_mel_ls= dividir_en_segmentos(normalizar_espectrograma_mel(drums_mel))
        bass_mel_ls = dividir_en_segmentos(normalizar_espectrograma_mel(bass_mel))
        others_mel_ls = dividir_en_segmentos(normalizar_espectrograma_mel(others_mel))
        vocals_mel_ls = dividir_en_segmentos(normalizar_espectrograma_mel(vocals_mel))

        #DRUMs espectrograma
        drums_dataloader = lista_to_dataloader(drums_mel_ls)
        drums_unet = procesar_unet(drums_dataloader, cargar_modelo(R'data\drums_entrenmiento\checkpoint_epoch_50.pth', device))
        unet_mel_drums = torch.concat(drums_unet, dim=2)

        #BASS espectrograma
        bass_dataloader = lista_to_dataloader(bass_mel_ls)
        bass_unet = procesar_unet(bass_dataloader, cargar_modelo(R'data\bass_entrenmiento\checkpoint_epoch_50.pth', device))
        unet_mel_bass = torch.concat(bass_unet, dim=2)

        #Vocal
        vocal_dataloader = lista_to_dataloader(vocals_mel_ls)
        vocal_unet = procesar_unet(vocal_dataloader, cargar_modelo(R'data\vocal_entrenmiento\checkpoint_epoch_50.pth', device))
        unet_mel_vocal = torch.concat(vocal_unet, dim=2)



def recorrer_archivos(directorio):
    # Expresiones regulares para los diferentes patrones de archivo
    patrones = [
        re.compile(r'^all_\d+\.wav$'),
        re.compile(r'^all_\d+\_bass.wav$'),
        re.compile(r'^all_\d+\_drums.wav$'),
        re.compile(r'^all_\d+\_vocals.wav$'),
        re.compile(r'^all_\d+\_other.wav$')
    ]
    
    # Lista para guardar las rutas de archivos por cada patrón
    archivos_filtrados = []
    rutas_archivo = []

    # Recorrer los archivos en el directorio
    for archivo in os.listdir(directorio):
        for patron in patrones:
            # Comprobar si el archivo coincide con el patrón
            if patron.match(archivo):
                # Obtener la ruta completa del archivo y agregarla a la lista
                ruta_completa = os.path.join(directorio, archivo)
                rutas_archivo.append(ruta_completa)
            if len(rutas_archivo)==5:
                archivos_filtrados.append(rutas_archivo)
                rutas_archivo = []
                break
            if len(rutas_archivo) > 5:
                raise ValueError(f"Error {archivo}")
            
        

    return archivos_filtrados


if __name__ == "__main__":
    #separar(R"C:\Users\Luis\Documents\DATASETS\musdb18hq\test\target\all_01.wav")

    out = recorrer_archivos(R"C:\Users\Luis\Documents\DATASETS\musdb18hq\test\target")
    print(out)










