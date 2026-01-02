device = 'cuda'
import musdb
import os
import csv
import museval
import numpy as np
import pandas as pd
import torch
from src.data.preprocesar import mel_spectrogram
from src.utils.datos import plot_mel_spectrogram
from src.utils.metricas import calculate_ssim, calcular_mse
from openunmix.predict import separate


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


def procesar_audio(audio, rate=44100):
    audio = torch.as_tensor(audio, dtype=torch.float32, device=device)
    return audio
    

def evaluar_SDR(modelo=None):
    mus = musdb.DB(root=None, download=True, subsets='test', is_wav=False)
    for track in mus:
        print(f"\n🔊 Canción: {track.name}")

        # Obtener el mix y su duración
        sample_rate = track.rate
        mix = track.audio
        mix = procesar_audio(mix)
        out = separate(mix, sample_rate, targets=["vocals", "drums", "bass", "other"], device=device)
        
        for key in out:
            out[key] = out[key][0].cpu().detach().numpy().T
        scores = museval.eval_mus_track(track, out)

        out = ""
        ls_datos = []
        ls_datos.append(track.name)
        for t in scores.scores["targets"]:
            out += t["name"].ljust(16) + "==> "
            for metric in ["SDR", "SIR", "ISR", "SAR"]:
                #out += (
                #    metric
                #    + ":"
                #    + "{:>8.3f}".format(
                #        scores.frames_agg(
                #            [float(f["metrics"][metric]) for f in t["frames"]]
                #        )
                #    )
                #    + "  "
                #)
                if metric == "SDR":
                    ls_datos.append(scores.frames_agg(
                            [float(f["metrics"][metric]) for f in t["frames"]]
                        ))
                    break
            out += "\n"
        
        print(f"vocals => {ls_datos[1]}")
        print(f"drums  => {ls_datos[2]}")
        print(f"bass   => {ls_datos[3]}")
        print(f"other  => {ls_datos[4]}")
        agregar_track_a_csv(ls_datos, 'sdr_metricas_open_un.csv', 'SDR')



from src.utils.datos import plot_two_mel_spectrograms
from src.utils.datos import normalizar_espectrograma_mel

def evaluar_SSIM(modelo=None):
    mus = musdb.DB(root=None, download=True, subsets='test', is_wav=False)

    for track in mus:
        ls_datos = [track.name]
        print(f"\n🔊 Canción: {track.name}")

        # Obtener el mix y su duración
        sample_rate = track.rate
        mix = track.audio
        mix = procesar_audio(mix)
        out = separate(mix, sample_rate, targets=["vocals", "drums", "bass", "other"], device=device)
        

        i_bass = torch.mean(out['bass'].squeeze(), dim=0, keepdim=True)
        i_bass = mel_spectrogram(i_bass)
        i_bass = normalizar_espectrograma_mel(i_bass)
        
        i_drums = torch.mean(out['drums'].squeeze(), dim=0, keepdim=True)
        i_drums = mel_spectrogram(i_drums)
        i_drums = normalizar_espectrograma_mel(i_drums)
        
        i_vocal = torch.mean(out['vocals'].squeeze(), dim=0, keepdim=True)
        i_vocal = mel_spectrogram(i_vocal)
        i_vocal = normalizar_espectrograma_mel(i_vocal)
        
        i_other = torch.mean(out['other'].squeeze(), dim=0, keepdim=True)
        i_other = mel_spectrogram(i_other)
        i_other = normalizar_espectrograma_mel(i_other)

        # Iterar sobre los stems (targets)
        for target_name, target in track.targets.items():
            print(target_name)
            audio = target.audio
            if target_name == "bass":
                audio = procesar_audio(audio)
                audio = torch.mean(audio.T, dim=0, keepdim=True)
                t_mel = mel_spectrogram(audio)
                t_mel = normalizar_espectrograma_mel(t_mel)
                ssim = calculate_ssim(t_mel, i_bass)
                ls_datos.append(ssim)
                
            elif target_name == "drums":
                audio = procesar_audio(audio)
                audio = torch.mean(audio.T, dim=0, keepdim=True)
                t_mel = mel_spectrogram(audio)
                t_mel = normalizar_espectrograma_mel(t_mel)
                ssim = calculate_ssim(t_mel, i_drums)
                ls_datos.append(ssim)

            elif target_name == "vocals":
                audio = procesar_audio(audio)
                audio = torch.mean(audio.T, dim=0, keepdim=True)
                t_mel = mel_spectrogram(audio)
                t_mel = normalizar_espectrograma_mel(t_mel)
                ssim = calculate_ssim(t_mel, i_vocal)
                ls_datos.append(ssim)

            elif target_name == "other":
                audio = procesar_audio(audio)
                audio = torch.mean(audio.T, dim=0, keepdim=True)
                t_mel = mel_spectrogram(audio)
                t_mel = normalizar_espectrograma_mel(t_mel)
                ssim = calculate_ssim(t_mel, i_other)
                ls_datos.append(ssim)
        agregar_track_a_csv(ls_datos, file_path='ssim_metricas_open_un.csv', metrica='SSIM')



def evaluar_MSE(modelo=None):
    mus = musdb.DB(root=None, download=True, subsets='test', is_wav=False)
    
    for track in mus:
        ls_datos = [track.name]
        print(f"\n🔊 Canción: {track.name}")

        # Obtener el mix y su duración
        sample_rate = track.rate
        mix = track.audio
        mix = procesar_audio(mix)
        out = separate(mix, sample_rate, targets=["vocals", "drums", "bass", "other"], device=device)
        

        i_bass = torch.mean(out['bass'].squeeze(), dim=0, keepdim=True)
        i_bass = mel_spectrogram(i_bass)
        i_bass = normalizar_espectrograma_mel(i_bass)
        
        i_drums = torch.mean(out['drums'].squeeze(), dim=0, keepdim=True)
        i_drums = mel_spectrogram(i_drums)
        i_drums = normalizar_espectrograma_mel(i_drums)
        
        i_vocal = torch.mean(out['vocals'].squeeze(), dim=0, keepdim=True)
        i_vocal = mel_spectrogram(i_vocal)
        i_vocal = normalizar_espectrograma_mel(i_vocal)
        
        i_other = torch.mean(out['other'].squeeze(), dim=0, keepdim=True)
        i_other = mel_spectrogram(i_other)
        i_other = normalizar_espectrograma_mel(i_other)

        # Iterar sobre los stems (targets)
        for target_name, target in track.targets.items():
            print(target_name)
            audio = target.audio
            if target_name == "bass":
                audio = procesar_audio(audio)
                audio = torch.mean(audio.T, dim=0, keepdim=True)
                t_mel = mel_spectrogram(audio)
                t_mel = normalizar_espectrograma_mel(t_mel)
                ssim = calcular_mse(t_mel, i_bass)
                ls_datos.append(ssim)
                
            elif target_name == "drums":
                audio = procesar_audio(audio)
                audio = torch.mean(audio.T, dim=0, keepdim=True)
                t_mel = mel_spectrogram(audio)
                t_mel = normalizar_espectrograma_mel(t_mel)
                ssim = calcular_mse(t_mel, i_drums)
                ls_datos.append(ssim)

            elif target_name == "vocals":
                audio = procesar_audio(audio)
                audio = torch.mean(audio.T, dim=0, keepdim=True)
                t_mel = mel_spectrogram(audio)
                t_mel = normalizar_espectrograma_mel(t_mel)
                ssim = calcular_mse(t_mel, i_vocal)
                ls_datos.append(ssim)

            elif target_name == "other":
                audio = procesar_audio(audio)
                audio = torch.mean(audio.T, dim=0, keepdim=True)
                t_mel = mel_spectrogram(audio)
                t_mel = normalizar_espectrograma_mel(t_mel)
                ssim = calcular_mse(t_mel, i_other)
                ls_datos.append(ssim)
        agregar_track_a_csv(ls_datos, 'mse_metricas_open_un.csv', 'MSE')


if __name__ == '__main__':
    #evaluar_SDR()
    #evaluar_SSIM()
    evaluar_MSE()
















