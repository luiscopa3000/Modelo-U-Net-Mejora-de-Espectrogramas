device = 'cuda'
import os
import torch
import torchaudio
import matplotlib.pyplot as plt
import torchaudio.transforms as T
import librosa
import numpy as np
import csv
from librosa.filters import mel as librosa_mel_fn


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def spectral_normalize_torch(magnitudes):
    return dynamic_range_compression_torch(magnitudes)



mel_basis_cache = {}
hann_window_cache = {}


def mel_spectrogram(
    y: torch.Tensor,
    n_fft: int = 2048,
    num_mels: int = 128,
    sampling_rate: int = 44100,
    hop_size: int = 512,
    win_size: int = 2048,
    fmin: int = 0,
    fmax: int = None,
    center: bool = False,
) -> torch.Tensor:
    """
    Calculate the mel spectrogram of an input signal.
    This function uses slaney norm for the librosa mel filterbank (using librosa.filters.mel) and uses Hann window for STFT (using torch.stft).

    Args:
        y (torch.Tensor): Input signal.
        n_fft (int): FFT size.
        num_mels (int): Number of mel bins.
        sampling_rate (int): Sampling rate of the input signal.
        hop_size (int): Hop size for STFT.
        win_size (int): Window size for STFT.
        fmin (int): Minimum frequency for mel filterbank.
        fmax (int): Maximum frequency for mel filterbank. If None, defaults to half the sampling rate (fmax = sr / 2.0) inside librosa_mel_fn
        center (bool): Whether to pad the input to center the frames. Default is False.

    Returns:
        torch.Tensor: Mel spectrogram.
    """
    if torch.min(y) < -1.0:
        print(f"[WARNING] Min value of input waveform signal is {torch.min(y)}")
    if torch.max(y) > 1.0:
        print(f"[WARNING] Max value of input waveform signal is {torch.max(y)}")

    device = y.device
    key = f"{n_fft}_{num_mels}_{sampling_rate}_{hop_size}_{win_size}_{fmin}_{fmax}_{device}"

    if key not in mel_basis_cache:
        mel = librosa_mel_fn(
            sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax
        )
        mel_basis_cache[key] = torch.from_numpy(mel).float().to(device)
        hann_window_cache[key] = torch.hann_window(win_size).to(device)

    mel_basis = mel_basis_cache[key]
    hann_window = hann_window_cache[key]

    padding = (n_fft - hop_size) // 2
    y = torch.nn.functional.pad(
        y.unsqueeze(1), (padding, padding), mode="reflect"
    ).squeeze(1)

    spec = torch.stft(
        y,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=hann_window,
        center=center,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=True,
    )
    spec = torch.sqrt(torch.view_as_real(spec).pow(2).sum(-1) + 1e-9)

    mel_spec = torch.matmul(mel_basis, spec)
    mel_spec = spectral_normalize_torch(mel_spec)

    return mel_spec


# Cargar el archivo de audio con sr=44100
#audio_path = R"C:\Users\Luis\Documents\DATASETS\musdb18hq\train\target\all_02_bass.wav"  # reemplaza con la ruta de tu archivo
#
#wav, sr = librosa.load(audio_path, sr=44100, mono=True) # wav is np.ndarray with shape [T_time] and values in [-1, 1]
#wav = torch.FloatTensor(wav).unsqueeze(0) # wav is FloatTensor with shape [B(1), T_time]
#mel = mel_spectrogram(wav)
#mel = mel[:,:,:4096]
#mel = mel.to(device)





folder_path = R'C:\Users\Luis\Documents\DATASETS\musdb18hq\train\target'


def dividir_espectrograma(espectrograma, tamaño_fragmento=128, solapamiento=0.5):
    _, num_mels, n_frames = espectrograma.shape
    paso = int(tamaño_fragmento * (1 - solapamiento))
    
    fragmentos = []
    for inicio in range(0, n_frames - tamaño_fragmento + 1, paso):
        fragmento = espectrograma[:, :, inicio:inicio + tamaño_fragmento].clone()
        fragmentos.append(fragmento)
    return fragmentos


def recorre_carpetas(input_folder,target_folder, nombre):
    target_list = []
    input_list = []
    for file_name in os.listdir(target_folder):
        if nombre in file_name:
            target_path = os.path.join(target_folder, file_name)
            input_path = os.path.join(input_folder, file_name)
            if not os.path.exists(input_path):
                raise ValueError(f"El archivo {file_name} no existe en la carpeta de inputs")
            target_list.append(target_path)
            input_list.append(input_path)

    if len(input_list) != len(target_list):
        raise ValueError("La cantidad de targets e inputs es diferente...")
    return input_list, target_list

import gzip
def guardar_tensores(input_tensors, nro_track, target_tensors, directorio, csv_path):
    # Verificar si las listas tienen el mismo tamaño
    if len(input_tensors) != len(target_tensors):
        raise ValueError("Las listas de tensores 'input' y 'target' deben tener la misma longitud.")

    # Verificar si el directorio existe, si no, crearlo
    if not os.path.exists(directorio):
        os.makedirs(directorio)

    # Verificar si el archivo CSV ya existe, si no, crearlo y escribir los encabezados
    csv_exists = os.path.exists(csv_path)
    with open(csv_path, mode='a', newline='') as csvfile:
        #fieldnames = ['file_name', 'input_shape', 'target_shape']
        fieldnames = ['ruta','file_name']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Escribir encabezados solo si el archivo no existe
        if not csv_exists:
            writer.writeheader()

        # Guardar cada par de tensores (input, target) en un archivo .pt
        for i, (input_tensor, target_tensor) in enumerate(zip(input_tensors, target_tensors)):
            # Nombre del archivo .pt
            ruta_guardado = os.path.join(directorio, f"track_{nro_track}_{i}.pt")
            
            # Crear un diccionario para almacenar los tensores
            data = {
                "input": input_tensor,
                "target": target_tensor
            }
            
            # Guardar el diccionario con los tensores
            torch.save(data, ruta_guardado)
            print(f"Archivo {i} guardado en {ruta_guardado}", end="\r")

            # Escribir en el archivo CSV los datos del archivo y sus dimensiones
            writer.writerow({
                'ruta': ruta_guardado,
                'file_name': f"track_{nro_track}_{i}.pt",
                #'input_shape': str(input_tensor.shape),
                #'target_shape': str(target_tensor.shape)
            })



def cambio_tono_velocidad(inputs_wav, sr = 44100, ls_tonos = [-4,-2,2,4], ls_rates =[0.8, 0.9, 1.1, 1.4]):
    #Wav_form tiene la forma(n_frames,)
    #Sr es el samplerates
        
    ls_wav = [inputs_wav]
    # Eliminar la dimensión extra
    nump_input = inputs_wav.squeeze()  # Esto cambiará la forma a [8843231]
    nump_input = nump_input.numpy() # comvierte a numpy

    for i in range(len(ls_tonos)):
        print(f" Haciendo pitch tono {ls_tonos[i]}...", end='\r')
        tono = librosa.effects.pitch_shift(nump_input, sr=sr, n_steps=ls_tonos[i])
        y_tono = torch.FloatTensor(tono).unsqueeze(0) # wav is FloatTensor with shape [B(1), T_time]
        ls_wav.append(y_tono)
        print(f" Cambiando velocidad a {ls_rates[i]}...", end='\r')
        velocidad = librosa.effects.time_stretch(y=tono, rate=ls_rates[i])
        y_velocidad = torch.FloatTensor(velocidad).unsqueeze(0) # wav is FloatTensor with shape [B(1), T_time]
        
        ls_wav.append(y_velocidad)
    return ls_wav


def procesar_audio(target_folder, input_folder, carpeta_out, nombre, csv_path):
    lista = recorre_carpetas(input_folder, target_folder, nombre)
    for i in range(len(lista[0])):
        inputs = lista[0][i]
        targets = lista[1][i]
        #print(inputs)
                
        inputs_wav, sr = librosa.load(inputs, sr=44100, mono=True) # wav is np.ndarray with shape [T_time] and values in [-1, 1]
        inputs_wav = torch.FloatTensor(inputs_wav).unsqueeze(0) # wav is FloatTensor with shape [B(1), T_time]
        
        targets_wav, sr = librosa.load(targets, sr=44100, mono=True) # wav is np.ndarray with shape [T_time] and values in [-1, 1]
        targets_wav = torch.FloatTensor(targets_wav).unsqueeze(0) # wav is FloatTensor with shape [B(1), T_time]
        
        ls_pitchs_inputs = cambio_tono_velocidad(inputs_wav=inputs_wav, sr=sr)
        ls_pitchs_target = cambio_tono_velocidad(inputs_wav=targets_wav, sr=sr)

        ls_parts_input =[]
        ls_parts_targ = []
        for i in range(len(ls_pitchs_target)):
            wav_inpt = ls_pitchs_inputs[i]
            wav_targ = ls_pitchs_target[i]

            mel_inputs = mel_spectrogram(wav_inpt)
            mel_targets = mel_spectrogram(wav_targ)
            #print(mel_inputs.shape)
            #torch.save(mel_inputs, "prueba.pt")
            #prueba = mel_inputs[:,:,:4096].clone()
            #torch.save(prueba, "prueba2.pt", _use_new_zipfile_serialization=True)
            #break
            if mel_inputs.shape != mel_targets.shape:
                raise ValueError(f"Las dimenciones son diferentes en {inputs}...")
        
            #mel_targets = (mel_targets - mel_targets.min()) / (mel_targets.max() - mel_targets.min())
            #mel_inputs = (mel_inputs - mel_inputs.min()) / (mel_inputs.max() - mel_inputs.min())
            parts_target = dividir_espectrograma(mel_targets, solapamiento=0.8)
            parts_inputs = dividir_espectrograma(mel_inputs, solapamiento=0.8)
            ls_parts_input += parts_inputs
            ls_parts_targ += parts_target


        nombre_archivo = os.path.basename(targets)
        nro_file = nombre_archivo.split('_')[1]

        guardar_tensores(input_tensors=ls_parts_input, target_tensors=ls_parts_targ, nro_track=nro_file, directorio=carpeta_out, csv_path=csv_path)
        print(f">> {nombre_archivo} guardado.")


def dividir_en_segmentos(tensor):
    # Verificamos que el tensor tenga la forma esperada
    if tensor.ndim != 3 or tensor.shape[0] != 1 or tensor.shape[1] != 128:
        raise ValueError("El tensor debe tener forma [1, 128, n_frames]")

    n_frames = tensor.shape[2]
    segmentos = []

    # Iteramos por pasos de 128
    for i in range(0, n_frames - 127, 128):
        segmento = tensor[:, :, i:i+128].clone()
        segmentos.append(segmento)

    return segmentos


def procesar_audio2(target_folder, input_folder, carpeta_out, nombre, csv_path):
    lista = recorre_carpetas(input_folder, target_folder, nombre)
    for i in range(len(lista[0])):
        inputs = lista[0][i]
        targets = lista[1][i]
        #print(inputs)
                
        inputs_wav, sr = librosa.load(inputs, sr=44100, mono=True) # wav is np.ndarray with shape [T_time] and values in [-1, 1]
        inputs_wav = torch.FloatTensor(inputs_wav).unsqueeze(0) # wav is FloatTensor with shape [B(1), T_time]
        
        targets_wav, sr = librosa.load(targets, sr=44100, mono=True) # wav is np.ndarray with shape [T_time] and values in [-1, 1]
        targets_wav = torch.FloatTensor(targets_wav).unsqueeze(0) # wav is FloatTensor with shape [B(1), T_time]
        
        mel_inputs = mel_spectrogram(inputs_wav)
        mel_targets = mel_spectrogram(targets_wav)
        #print(mel_inputs.shape)
        #torch.save(mel_inputs, "prueba.pt")
        #prueba = mel_inputs[:,:,:4096].clone()
        #torch.save(prueba, "prueba2.pt", _use_new_zipfile_serialization=True)
        #break
        if mel_inputs.shape != mel_targets.shape:
            raise ValueError(f"Las dimenciones son diferentes en {inputs}...")
    
        #mel_targets = (mel_targets - mel_targets.min()) / (mel_targets.max() - mel_targets.min())
        #mel_inputs = (mel_inputs - mel_inputs.min()) / (mel_inputs.max() - mel_inputs.min())
        parts_target = dividir_espectrograma(mel_targets, solapamiento=0.5)
        parts_inputs = dividir_espectrograma(mel_inputs, solapamiento=0.5)


        nombre_archivo = os.path.basename(targets)
        nro_file = nombre_archivo.split('_')[1]

        guardar_tensores(input_tensors=parts_inputs, target_tensors=parts_target, nro_track=nro_file, directorio=carpeta_out, csv_path=csv_path)
        print(f">> {nombre_archivo} guardado.")



def procesar_audio3(target_folder, input_folder, carpeta_out, nombre, csv_path):
    lista = recorre_carpetas(input_folder, target_folder, nombre)
    for i in range(len(lista[0])):
        inputs = lista[0][i]
        targets = lista[1][i]
        #print(inputs)
                
        inputs_wav, sr = librosa.load(inputs, sr=44100, mono=True) # wav is np.ndarray with shape [T_time] and values in [-1, 1]
        inputs_wav = torch.FloatTensor(inputs_wav).unsqueeze(0) # wav is FloatTensor with shape [B(1), T_time]
        
        targets_wav, sr = librosa.load(targets, sr=44100, mono=True) # wav is np.ndarray with shape [T_time] and values in [-1, 1]
        targets_wav = torch.FloatTensor(targets_wav).unsqueeze(0) # wav is FloatTensor with shape [B(1), T_time]
        
        mel_inputs = mel_spectrogram(inputs_wav)
        mel_targets = mel_spectrogram(targets_wav)
        #print(mel_inputs.shape)
        #torch.save(mel_inputs, "prueba.pt")
        #prueba = mel_inputs[:,:,:4096].clone()
        #torch.save(prueba, "prueba2.pt", _use_new_zipfile_serialization=True)
        #break
        if mel_inputs.shape != mel_targets.shape:
            raise ValueError(f"Las dimenciones son diferentes en {inputs}...")
    
        #mel_targets = (mel_targets - mel_targets.min()) / (mel_targets.max() - mel_targets.min())
        #mel_inputs = (mel_inputs - mel_inputs.min()) / (mel_inputs.max() - mel_inputs.min())
        parts_target = dividir_en_segmentos(mel_targets)
        parts_inputs = dividir_en_segmentos(mel_inputs)


        nombre_archivo = os.path.basename(targets)
        nro_file = nombre_archivo.split('_')[1]

        guardar_tensores(input_tensors=parts_inputs, target_tensors=parts_target, nro_track=nro_file, directorio=carpeta_out, csv_path=csv_path)
        print(f">> {nombre_archivo} guardado.")




if __name__ == "__main__":
    nombre = 'drums'

    target_folder = R'C:\Users\Luis\Documents\DATASETS\musdb18hq\test\target'
    input_folder = R'C:\Users\Luis\Documents\DATASETS\musdb18hq\test\input'

    carpeta_out = R'C:\Users\Luis\Desktop\final_proy\env\src\data\precomputed\valid'
    csv_path = R'C:\Users\Luis\Desktop\final_proy\env\src\data\precomputed\valid.csv'

    procesar_audio3(target_folder, input_folder, nombre=nombre, carpeta_out=carpeta_out,csv_path=csv_path)



