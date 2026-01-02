import os
import librosa
import torch
import csv

from src.data.preprocesar import mel_spectrogram
from src.utils.metricas import calculate_ssim, calcular_mse, SDR_calcular
from src.data.preprocesar import dividir_en_segmentos
from src.utils.spectrograms import normalizar_espectrograma_mel, split_spectrogram, nivelar_mel


def agregar_track_a_csv(track_data, file_path="tracks_data.csv", metrica= 'MSE'):
    """
    Función que guarda o agrega un track con sus datos de SDR en un archivo CSV.
    
    :param track_data: Lista con el nombre del track y los SDR de vocals, drums, bass, y others.
    :param file_path: Ruta al archivo CSV donde se guardarán los datos (por defecto 'tracks_data.csv').
    """
    # Verificar si el archivo CSV existe
    file_exists = os.path.exists(file_path)

    # Cabeceras para el CSV
    headers = [f"{metrica} Vocals", f"{metrica} Drums", f"{metrica} Bass", f"{metrica} Others"]

    # Abrir el archivo CSV en modo append ('a'), si no existe, creará el archivo
    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)

        # Si el archivo no existe, escribir las cabeceras
        if not file_exists:
            writer.writerow(headers)

        # Escribir la nueva fila con los datos
        writer.writerow(track_data)

    print(f"Datos añadidos al archivo {file_path}.")



def alinear_tensores(tensor1, tensor2):
    """
    Recorta dos tensores en la dimensión 2 (temporal) para que tengan la misma longitud.
    
    Args:
    tensor1, tensor2 (torch.Tensor): Tensores de forma [1, N].
    
    Returns:
    tensor1_recortado, tensor2_recortado: Tensores recortados con la misma longitud temporal.
    """
    min_frames = min(tensor1.shape[1], tensor2.shape[1])
    tensor1_recortado = tensor1[:, :min_frames]
    tensor2_recortado = tensor2[:, :min_frames]
    return tensor1_recortado, tensor2_recortado

def construir_ruta(i, inst):
    indice = i if len(str(i)) != 1 else '0'+str(i)
    archivo = f"all_{indice}_{inst}.wav"
    return archivo

def cargar_audios(wav_path, wav_muestra):
    # cargar archivo wav y calcular el espectrograma mel
    #wav_path = R"C:\Users\Luis\Documents\DATASETS\musdb18hq\train\target\all_01_bass.wav"
    wav_s, sr = librosa.load(wav_path, sr=44100, mono=True) # wav es un np.ndarray con forma [T_time] y valores en [-1, 1]
    wav_s = torch.FloatTensor(wav_s).unsqueeze(0) # wav es un FloatTensor con forma [B(1), T_time]

    wav_m, sr = librosa.load(wav_muestra, sr=44100, mono=True) # wav es un np.ndarray con forma [T_time] y valores en [-1, 1]
    wav_m = torch.FloatTensor(wav_m).unsqueeze(0) # wav es un FloatTensor con forma [B(1), T_time]
    return alinear_tensores(wav_s, wav_m)

def procesar_mse(r_spl, r_mue):
    ws, wm = cargar_audios(r_spl, r_mue)
    ms = normalizar_espectrograma_mel(mel_spectrogram(ws))
    mm = normalizar_espectrograma_mel(mel_spectrogram(wm))
    return calcular_mse(ms, mm)


def evaluar_mse(ruta_spleeter, ruta_muestra):
    n = 50
    ruta = ''
    ls_data = []
    for i in range(1,n+1):
        print("Track",i)
        rs_bass = os.path.join(ruta_spleeter, construir_ruta(i, 'bass'))
        rt_bass = os.path.join(ruta_muestra, construir_ruta(i, 'bass'))
        mse_bass = procesar_mse(rs_bass, rt_bass)

        rs_drums = os.path.join(ruta_spleeter, construir_ruta(i, 'drums'))
        rt_drums = os.path.join(ruta_muestra, construir_ruta(i, 'drums'))
        mse_drums = procesar_mse(rs_drums, rt_drums)

        rs_vocals = os.path.join(ruta_spleeter, construir_ruta(i, 'vocals'))
        rt_vocals = os.path.join(ruta_muestra, construir_ruta(i, 'vocals'))
        mse_vocals = procesar_mse(rs_vocals, rt_vocals)
        
        rs_other = os.path.join(ruta_spleeter, construir_ruta(i, 'other'))
        rt_other = os.path.join(ruta_muestra, construir_ruta(i, 'other'))
        mse_other = procesar_mse(rs_other, rt_other)

        ls = [mse_bass, mse_drums, mse_vocals, mse_other]
        agregar_track_a_csv(ls, 'mse_metricas_spleeter.csv', 'MSE')


def procesar_ssim(r_spl, r_mue):
    ws, wm = cargar_audios(r_spl, r_mue)
    ms = normalizar_espectrograma_mel(mel_spectrogram(ws))
    mm = normalizar_espectrograma_mel(mel_spectrogram(wm))
    return calculate_ssim(ms, mm)


def evaluar_ssim(ruta_spleeter, ruta_muestra):
    n = 50
    ruta = ''
    ls_data = []
    for i in range(1,n+1):
        print("Track",i)
        rs_bass = os.path.join(ruta_spleeter, construir_ruta(i, 'bass'))
        rt_bass = os.path.join(ruta_muestra, construir_ruta(i, 'bass'))
        ssim_bass = procesar_ssim(rs_bass, rt_bass)

        rs_drums = os.path.join(ruta_spleeter, construir_ruta(i, 'drums'))
        rt_drums = os.path.join(ruta_muestra, construir_ruta(i, 'drums'))
        ssim_drums = procesar_ssim(rs_drums, rt_drums)

        rs_vocals = os.path.join(ruta_spleeter, construir_ruta(i, 'vocals'))
        rt_vocals = os.path.join(ruta_muestra, construir_ruta(i, 'vocals'))
        ssim_vocals = procesar_ssim(rs_vocals, rt_vocals)
        
        rs_other = os.path.join(ruta_spleeter, construir_ruta(i, 'other'))
        rt_other = os.path.join(ruta_muestra, construir_ruta(i, 'other'))
        ssim_other = procesar_ssim(rs_other, rt_other)

        ls = [ssim_bass, ssim_drums, ssim_vocals, ssim_other]
        agregar_track_a_csv(ls, 'ssim_metricas_spleeter.csv', 'SSIM')






def procesar_sdr(r_spl, r_mue):
    ws, wm = cargar_audios(r_spl, r_mue)
    return SDR_calcular(ws, wm)


def evaluar_sdr(ruta_spleeter, ruta_muestra):
    n = 50
    ruta = ''
    ls_data = []
    for i in range(1,n+1):
        print("Track",i)
        rs_bass = os.path.join(ruta_spleeter, construir_ruta(i, 'bass'))
        rt_bass = os.path.join(ruta_muestra, construir_ruta(i, 'bass'))
        ssim_bass = procesar_sdr(rs_bass, rt_bass)

        rs_drums = os.path.join(ruta_spleeter, construir_ruta(i, 'drums'))
        rt_drums = os.path.join(ruta_muestra, construir_ruta(i, 'drums'))
        ssim_drums = procesar_sdr(rs_drums, rt_drums)

        rs_vocals = os.path.join(ruta_spleeter, construir_ruta(i, 'vocals'))
        rt_vocals = os.path.join(ruta_muestra, construir_ruta(i, 'vocals'))
        ssim_vocals = procesar_sdr(rs_vocals, rt_vocals)
        
        rs_other = os.path.join(ruta_spleeter, construir_ruta(i, 'other'))
        rt_other = os.path.join(ruta_muestra, construir_ruta(i, 'other'))
        ssim_other = procesar_sdr(rs_other, rt_other)

        ls = [ssim_bass, ssim_drums, ssim_vocals, ssim_other]
        agregar_track_a_csv(ls, 'sdr_metricas_spleeter.csv', 'SDR')



#evaluar_mse(R'C:\Users\Luis\Documents\DATASETS\spleeter', R'C:\Users\Luis\Documents\DATASETS\musdb18hq\test\target')
#evaluar_ssim(R'C:\Users\Luis\Documents\DATASETS\spleeter', R'C:\Users\Luis\Documents\DATASETS\musdb18hq\test\target')
evaluar_sdr(R'C:\Users\Luis\Documents\DATASETS\spleeter', R'C:\Users\Luis\Documents\DATASETS\musdb18hq\test\target')



