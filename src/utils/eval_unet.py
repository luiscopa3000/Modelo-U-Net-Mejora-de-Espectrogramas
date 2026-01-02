import musdb

device = 'cuda'
import os
import csv
import museval
import numpy as np
import pandas as pd
import torch
from src.models.demucsm import cargar_demucs, separar_eval_demucs
from src.data.preprocesar import mel_spectrogram
from src.utils.datos import plot_mel_spectrogram
from src.utils.metricas import calculate_ssim, calcular_mse, SDR_calcular
from src.scripts.separador import Refinador
from src.utils.waveforms import nivelar_waveforms


from src.utils.datos import plot_two_mel_spectrograms
from src.utils.datos import normalizar_espectrograma_mel

def recortar_tensores(tensor1, tensor2):
    # Obtenemos el número de frames de ambos tensores
    n_frames1 = tensor1.shape[1]
    n_frames2 = tensor2.shape[1]
    
    # Calculamos el número mínimo de frames
    n_frames_min = min(n_frames1, n_frames2)
    
    # Recortamos ambos tensores
    tensor1_recortado = tensor1[:, :n_frames_min]
    tensor2_recortado = tensor2[:, :n_frames_min]
    
    return tensor1_recortado, tensor2_recortado

def cortar_espectrograma(tensor1, tensor2):
    n_frames_min = min(tensor1.shape[2], tensor2.shape[2])
    return tensor1[:, :, :n_frames_min], tensor2[:, :, :n_frames_min]

def agregar_track_a_csv(track_data, file_path="tracks_data.csv", metrica= 'SDR'):
    """
    Función que guarda o agrega un track con sus datos de SDR en un archivo CSV.
    
    :param track_data: Lista con el nombre del track y los SDR de vocals, drums, bass, y others.
    :param file_path: Ruta al archivo CSV donde se guardarán los datos (por defecto 'tracks_data.csv').
    """
    # Verificar si el archivo CSV existe
    file_exists = os.path.exists(file_path)

    # Cabeceras para el CSV
    headers = ["Track Name", f"{metrica} Vocals", f"{metrica} Drums", f"{metrica} Bass", f"{metrica} Others"]

    # Abrir el archivo CSV en modo append ('a'), si no existe, creará el archivo
    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)

        # Si el archivo no existe, escribir las cabeceras
        if not file_exists:
            writer.writerow(headers)

        # Escribir la nueva fila con los datos
        writer.writerow(track_data)

    print(f"Datos añadidos al archivo {file_path}.")


def procesar_audio(audio):
    audio = torch.as_tensor(audio, dtype=torch.float32, device=device)
    audio = audio.transpose(0, 1)  # Cambia de [300032, 1] a [1, 300032]
    return audio
    

def evaluar_SDR(modelo=None):
    mus = musdb.DB(root=None, download=True, subsets='test', is_wav=False)
    
    for track in mus:
        ls_datos = [track.name]
        print(f"\n🔊 Canción: {track.name}")

        # Obtener el mix y su duración
        mix = track.audio
        mix = procesar_audio(mix)
        sample_rate = track.rate
        out = modelo.get_mel(mix)
        i_bass = out[1]
        i_bass = normalizar_espectrograma_mel(i_bass)

        i_drums = out[0]
        i_drums = normalizar_espectrograma_mel(i_drums)

        i_vocal = out[3]
        i_vocal = normalizar_espectrograma_mel(i_vocal)

        i_other = out[2]
        i_other = normalizar_espectrograma_mel(i_other)

        # Iterar sobre los stems (targets)
        for target_name, target in track.targets.items():
            print(target_name)
            audio = target.audio
            if target_name == "bass":
                audio = procesar_audio(audio)
                audio = torch.mean(audio, dim=0, keepdim=True)
                t_mel = mel_spectrogram(audio)
                t_mel = normalizar_espectrograma_mel(t_mel)
                t_mel, i_bass = cortar_espectrograma(t_mel, i_bass)
                sdr = SDR_calcular(t_mel, i_bass)
                ls_datos.append(sdr)
                
            elif target_name == "drums":
                audio = procesar_audio(audio)
                audio = torch.mean(audio, dim=0, keepdim=True)
                t_mel = mel_spectrogram(audio)
                t_mel = normalizar_espectrograma_mel(t_mel)
                t_mel, i_drums = cortar_espectrograma(t_mel, i_drums)
                sdr = SDR_calcular(t_mel, i_drums)
                ls_datos.append(sdr)

            elif target_name == "vocals":
                audio = procesar_audio(audio)
                audio = torch.mean(audio, dim=0, keepdim=True)
                t_mel = mel_spectrogram(audio)
                t_mel = normalizar_espectrograma_mel(t_mel)
                t_mel, i_vocal = cortar_espectrograma(t_mel, i_vocal)
                sdr = SDR_calcular(t_mel, i_vocal)
                ls_datos.append(sdr)

            elif target_name == "other":
                audio = procesar_audio(audio)
                audio = torch.mean(audio, dim=0, keepdim=True)
                t_mel = mel_spectrogram(audio)
                t_mel = normalizar_espectrograma_mel(t_mel)
                t_mel, i_other = cortar_espectrograma(t_mel, i_other)
                sdr = SDR_calcular(t_mel, i_other)
                ls_datos.append(sdr)
        agregar_track_a_csv(ls_datos, file_path=R'sdr_metricas_unet.csv', metrica='SDR')




def evaluar_SSIM(modelo=None):
    mus = musdb.DB(root=None, download=True, subsets='test', is_wav=False)
    
    for track in mus:
        ls_datos = [track.name]
        print(f"\n🔊 Canción: {track.name}")

        # Obtener el mix y su duración
        mix = track.audio
        mix = procesar_audio(mix)
        sample_rate = track.rate
        out = modelo.get_mel(mix)
        i_bass = out[1]
        i_bass = normalizar_espectrograma_mel(i_bass)

        i_drums = out[0]
        i_drums = normalizar_espectrograma_mel(i_drums)

        i_vocal = out[3]
        i_vocal = normalizar_espectrograma_mel(i_vocal)

        i_other = out[2]
        i_other = normalizar_espectrograma_mel(i_other)

        # Iterar sobre los stems (targets)
        for target_name, target in track.targets.items():
            print(target_name)
            audio = target.audio
            if target_name == "bass":
                audio = procesar_audio(audio)
                audio = torch.mean(audio, dim=0, keepdim=True)
                t_mel = mel_spectrogram(audio)
                t_mel = normalizar_espectrograma_mel(t_mel)
                t_mel, i_bass = cortar_espectrograma(t_mel, i_bass)
                ssim = calculate_ssim(t_mel, i_bass)
                ls_datos.append(ssim)
                
            elif target_name == "drums":
                audio = procesar_audio(audio)
                audio = torch.mean(audio, dim=0, keepdim=True)
                t_mel = mel_spectrogram(audio)
                t_mel = normalizar_espectrograma_mel(t_mel)
                t_mel, i_drums = cortar_espectrograma(t_mel, i_drums)
                ssim = calculate_ssim(t_mel, i_drums)
                ls_datos.append(ssim)

            elif target_name == "vocals":
                audio = procesar_audio(audio)
                audio = torch.mean(audio, dim=0, keepdim=True)
                t_mel = mel_spectrogram(audio)
                t_mel = normalizar_espectrograma_mel(t_mel)
                t_mel, i_vocal = cortar_espectrograma(t_mel, i_vocal)
                ssim = calculate_ssim(t_mel, i_vocal)
                ls_datos.append(ssim)

            elif target_name == "other":
                audio = procesar_audio(audio)
                audio = torch.mean(audio, dim=0, keepdim=True)
                t_mel = mel_spectrogram(audio)
                t_mel = normalizar_espectrograma_mel(t_mel)
                t_mel, i_other = cortar_espectrograma(t_mel, i_other)
                ssim = calculate_ssim(t_mel, i_other)
                ls_datos.append(ssim)
        agregar_track_a_csv(ls_datos, file_path=R'ssim_metricas_unet.csv', metrica='SSIM')



def evaluar_MSE(modelo=None):
    mus = musdb.DB(root=None, download=True, subsets='test', is_wav=False)
    
    for track in mus:
        ls_datos = [track.name]
        print(f"\n🔊 Canción: {track.name}")

        # Obtener el mix y su duración
        mix = track.audio
        mix = procesar_audio(mix)
        sample_rate = track.rate
        out = modelo.get_mel(mix)
        i_bass = out[1]
        i_bass = normalizar_espectrograma_mel(i_bass)

        i_drums = out[0]
        i_drums = normalizar_espectrograma_mel(i_drums)

        i_vocal = out[3]
        i_vocal = normalizar_espectrograma_mel(i_vocal)

        i_other = out[2]
        i_other = normalizar_espectrograma_mel(i_other)

        # Iterar sobre los stems (targets)
        for target_name, target in track.targets.items():
            print(target_name)
            audio = target.audio
            if target_name == "bass":
                audio = procesar_audio(audio)
                audio = torch.mean(audio, dim=0, keepdim=True)
                t_mel = mel_spectrogram(audio)
                t_mel = normalizar_espectrograma_mel(t_mel)
                t_mel, i_bass = cortar_espectrograma(t_mel, i_bass)
                ssim = calcular_mse(t_mel, i_bass)
                ls_datos.append(ssim)
                
            elif target_name == "drums":
                audio = procesar_audio(audio)
                audio = torch.mean(audio, dim=0, keepdim=True)
                t_mel = mel_spectrogram(audio)
                t_mel = normalizar_espectrograma_mel(t_mel)
                t_mel, i_drums = cortar_espectrograma(t_mel, i_drums)
                ssim = calcular_mse(t_mel, i_drums)
                ls_datos.append(ssim)

            elif target_name == "vocals":
                audio = procesar_audio(audio)
                audio = torch.mean(audio, dim=0, keepdim=True)
                t_mel = mel_spectrogram(audio)
                t_mel = normalizar_espectrograma_mel(t_mel)
                t_mel, i_vocal = cortar_espectrograma(t_mel, i_vocal)
                ssim = calcular_mse(t_mel, i_vocal)
                ls_datos.append(ssim)

            elif target_name == "other":
                audio = procesar_audio(audio)
                audio = torch.mean(audio, dim=0, keepdim=True)
                t_mel = mel_spectrogram(audio)
                t_mel = normalizar_espectrograma_mel(t_mel)
                t_mel, i_other = cortar_espectrograma(t_mel, i_other)
                ssim = calcular_mse(t_mel, i_other)
                ls_datos.append(ssim)
        agregar_track_a_csv(ls_datos, R'mse_metricas_unet.csv', 'MSE')



if __name__ == '__main__':
    modelo = Refinador()
    #evaluar_SDR(modelo)
    #evaluar_SSIM(modelo)

    evaluar_MSE(modelo)



