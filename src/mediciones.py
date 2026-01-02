import os
import re
import torch
import torchaudio
from torch.utils.data import DataLoader
from tqdm import tqdm
import librosa
import csv

from src.models.unet import UNet, cargar_modelo
from src.models.disc import Discriminator
from src.models.demucsm import cargar_demucs, separar_demucs


from src.data.preprocesar import mel_spectrogram
from src.data.preprocesar import dividir_en_segmentos
from src.utils.datos import plot_mel_spectrogram
from src.utils.datos import plot_two_mel_spectrograms
from src.utils.metricas import SDR_calcular, SSIM_media, calcular_mse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def guardar_resultados_csv(dem_sdr, dem_ssim, dem_mse, unet_sdr, unet_ssim, unet_mse, archivo_csv='resultados.csv'):
    # Comprobamos si el archivo CSV ya existe
    existe_archivo = os.path.isfile(archivo_csv)

    # Definir los datos a guardar
    resultados = [
        ["dem_sdr", "dem_ssim", "dem_mse", "unet_sdr", "unet_ssim", "unet_mse"],  # encabezado
        [dem_sdr, dem_ssim, dem_mse, unet_sdr, unet_ssim, unet_mse]                # los resultados
    ]
    
    # Abrir el archivo en modo append (añadir al final)
    with open(archivo_csv, mode='a', newline='') as f:
        writer = csv.writer(f)
        
        # Si el archivo no existe, escribir el encabezado
        if not existe_archivo:
            writer.writerow(resultados[0])  # encabezados
        
        # Escribir los resultados
        writer.writerow(resultados[1])
        
    print(f"Resultados guardados en {archivo_csv}")


def lista_to_dataloader(lista):
    dataloader = DataLoader(
        lista,
        batch_size=32,
        shuffle=False
    )
    return dataloader


# Función para cargar el modelo entrenado
def mod_peso(modelo, path_modelo: str, device: torch.device = None):
    checkpoint = torch.load(path_modelo, map_location=device)
    modelo.load_state_dict(checkpoint['generator_state_dict'])
    return modelo



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


def procesar_unet(data, model):
    ls_temp = []
    with torch.inference_mode():
        for mel in tqdm(data):
            mel = mel.to(device='cuda')
            wav_gen = model(mel)
            ls_temp.extend(list(wav_gen))

    return ls_temp



def medir(all, bass, drums, vocals, others, demucs, unet):
    print(all)
    out = separar_demucs(audio_path=all, model=demucs)
    d_drums = out[0]
    d_bass = out[1]
    d_others = out[2]
    d_vocals = out[3]
    
    #Espectrogramas mel
    d_drums_mel = normalizar_espectrograma_mel(mel_spectrogram(d_drums))
    d_bass_mel = normalizar_espectrograma_mel(mel_spectrogram(d_bass))
    d_others_mel = normalizar_espectrograma_mel(mel_spectrogram(d_others))
    d_vocals_mel = normalizar_espectrograma_mel(mel_spectrogram(d_vocals))


    #Espectrogramas mel target
    wav_drums, sr = librosa.load(drums, sr=44100, mono=True) # wav es un np.ndarray con forma [T_time] y valores en [-1, 1]
    wav_drums = torch.FloatTensor(wav_drums).unsqueeze(0) # wav es un FloatTensor con forma [B(1), T_time]
    drums_mel = normalizar_espectrograma_mel(mel_spectrogram(wav_drums))

    wav_bass, sr = librosa.load(bass, sr=44100, mono=True) # wav es un np.ndarray con forma [T_time] y valores en [-1, 1]
    wav_bass = torch.FloatTensor(wav_bass).unsqueeze(0) # wav es un FloatTensor con forma [B(1), T_time]
    bass_mel = normalizar_espectrograma_mel(mel_spectrogram(wav_bass))

    
    wav_vocal, sr = librosa.load(vocals, sr=44100, mono=True) # wav es un np.ndarray con forma [T_time] y valores en [-1, 1]
    wav_vocal = torch.FloatTensor(wav_vocal).unsqueeze(0) # wav es un FloatTensor con forma [B(1), T_time]
    vocals_mel = normalizar_espectrograma_mel(mel_spectrogram(wav_vocal))

    #others_mel = normalizar_espectrograma_mel(mel_spectrogram(others))


    #Espectrogramas mel por chunks
    drums_mel_ls= dividir_en_segmentos(d_drums_mel)
    bass_mel_ls = dividir_en_segmentos(d_bass_mel)
    others_mel_ls = dividir_en_segmentos(d_others_mel)
    vocals_mel_ls = dividir_en_segmentos(d_vocals_mel)


    #DRUMs espectrograma
    drums_dataloader = lista_to_dataloader(drums_mel_ls)
    drums_unet = procesar_unet(drums_dataloader, mod_peso(unet,R'data\drums_entrenmiento\checkpoint_epoch_50.pth', device))
    unet_mel_drums = torch.concat(drums_unet, dim=2)
    #plot_two_mel_spectrograms(unet_mel_drums.cpu(), drums_mel.cpu())
    drums_mel = drums_mel[:,:,:unet_mel_drums.shape[2]]
    d_drums_mel = d_drums_mel[:,:,:unet_mel_drums.shape[2]]

    # Mover los tensores a la misma GPU o CPU
    d_drums_mel = d_drums_mel.to(device)
    unet_mel_drums = unet_mel_drums.to(device)
    drums_mel = drums_mel.to(device)

    dem_sdr = SDR_calcular(d_drums_mel, drums_mel)
    dem_ssim = SSIM_media(d_drums_mel, drums_mel)
    dem_mse = calcular_mse(d_drums_mel, drums_mel)


    unet_sdr = SDR_calcular(unet_mel_drums, drums_mel)
    unet_ssim = SSIM_media(unet_mel_drums, drums_mel)
    unet_mse = calcular_mse(unet_mel_drums, drums_mel)
    #guardar_resultados_csv(dem_sdr, dem_ssim, dem_mse, unet_sdr, unet_ssim, unet_mse, 'drums.csv')



    #BASS espectrograma
    bass_dataloader = lista_to_dataloader(bass_mel_ls)
    bass_unet = procesar_unet(bass_dataloader, mod_peso(unet,R'data\bass_entrenmiento\checkpoint_epoch_50.pth', device))
    unet_mel_bass = torch.concat(bass_unet, dim=2)
    #plot_two_mel_spectrograms(unet_mel_bass.cpu(), bass_mel.cpu())
    bass_mel = bass_mel[:,:,:unet_mel_bass.shape[2]]
    d_bass_mel = d_bass_mel[:,:,:unet_mel_bass.shape[2]]

    # Mover los tensores a la misma GPU o CPU
    d_bass_mel = d_bass_mel.to(device)
    unet_mel_bass = unet_mel_bass.to(device)
    bass_mel = bass_mel.to(device)

    dem_sdr = SDR_calcular(d_bass_mel, bass_mel)
    dem_ssim = SSIM_media(d_bass_mel, bass_mel)
    dem_mse = calcular_mse(d_bass_mel, bass_mel)


    unet_sdr = SDR_calcular(unet_mel_bass, bass_mel)
    unet_ssim = SSIM_media(unet_mel_bass, bass_mel)
    unet_mse = calcular_mse(unet_mel_bass, bass_mel)
    #guardar_resultados_csv(dem_sdr, dem_ssim, dem_mse, unet_sdr, unet_ssim, unet_mse, 'bass.csv')



    #Vocal
    vocal_dataloader = lista_to_dataloader(vocals_mel_ls)
    vocal_unet = procesar_unet(vocal_dataloader, mod_peso(unet,R'data\vocal_entrenmiento\checkpoint_epoch_51.pth', device))
    unet_mel_vocal = torch.concat(vocal_unet, dim=2)
    #plot_two_mel_spectrograms(unet_mel_vocal.cpu(), vocals_mel.cpu())
    
    vocals_mel = vocals_mel[:,:,:unet_mel_vocal.shape[2]]
    d_vocals_mel = d_vocals_mel[:,:,:unet_mel_vocal.shape[2]]
    
    # Mover los tensores a la misma GPU o CPU
    d_vocals_mel = d_vocals_mel.to(device)
    unet_mel_vocal = unet_mel_vocal.to(device)
    vocals_mel = vocals_mel.to(device)

    dem_sdr = SDR_calcular(d_vocals_mel, vocals_mel)
    dem_ssim = SSIM_media(d_vocals_mel, vocals_mel)
    dem_mse = calcular_mse(d_vocals_mel, vocals_mel)


    unet_sdr = SDR_calcular(unet_mel_vocal, vocals_mel)
    unet_ssim = SSIM_media(unet_mel_vocal, vocals_mel)
    unet_mse = calcular_mse(unet_mel_vocal, vocals_mel)
    guardar_resultados_csv(dem_sdr, dem_ssim, dem_mse, unet_sdr, unet_ssim, unet_mse, 'vocal.csv')








if __name__ == "__main__":
    #separar(R"C:\Users\Luis\Documents\DATASETS\musdb18hq\test\target\all_01.wav")
    m_dem = cargar_demucs()
    unet = cargar_modelo(path_modelo=R'data\bass_entrenmiento\checkpoint_epoch_50.pth', device='cuda')

    out = recorrer_archivos(R"C:\Users\Luis\Documents\DATASETS\musdb18hq\test\target")
    for ru in out:
        medir(ru[0], ru[1], ru[2], ru[3], ru[4], m_dem, unet)





















