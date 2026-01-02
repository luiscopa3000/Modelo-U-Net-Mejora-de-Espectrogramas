import os
from pathlib import Path
import demucs
from pydub import AudioSegment
import torch
import typing as tp
import torchaudio as ta

from src.utils.waveforms import convertir_a_mono

def convert_to_mono(directory):
    """Convierte todos los archivos de audio en un directorio a mono"""
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith((".wav", ".mp3", ".flac", ".ogg")):
                file_path = os.path.join(root, file)
                audio = AudioSegment.from_file(file_path)
                mono_audio = audio.set_channels(1)
                mono_audio.export(file_path, format="wav")  # Sobrescribe el archivo en mono
                print(f"Converted {file} to mono.")

class SepararAudio:
    def __init__(self, model="htdemucs", extensions=None, 
                 mp3=False, mp3_rate=320, float32=False, int24=False, 
                 in_path='audio/input', out_path='audio/output'):
        self.model = model
        self.extensions = extensions or ["mp3", "wav", "ogg", "flac"]
        self.mp3 = mp3  # Cambiar a False por defecto
        self.mp3_rate = mp3_rate
        self.float32 = float32
        self.int24 = int24
        self.in_path = in_path
        self.out_path = out_path

        # Crear directorio de salida si no existe
        Path(self.out_path).mkdir(parents=True, exist_ok=True)

    # Método para buscar un solo archivo de audio en el directorio
    def find_file(self):
        for file in Path(self.in_path).iterdir():
            if file.suffix.lower().lstrip(".") in self.extensions:
                return file
        return None

    # Método para separar el audio usando el modelo de Demucs
    def separate(self):
        file = self.find_file()
        if file is None:
            print(f"No valid audio file found in {self.in_path}")
            return None

        print(f"Going to separate the file: {file}")
        
        # Construir el comando para Demucs como una lista de argumentos
        cmd = [
            "-o", str(self.out_path), 
            "-n", self.model
        ]
        
        # Eliminar la opción MP3 si no se necesita
        if self.mp3:
            cmd += ["--mp3", f"--mp3-bitrate={self.mp3_rate}"]

        if self.float32:
            cmd += ["--float32"]
        if self.int24:
            cmd += ["--int24"]

        # Incluir el archivo de audio
        cmd.append(str(file))

        # Ejecutar demucs.separate.main() pasando los argumentos
        print(f"With command: {cmd}")
        demucs.separate.main(cmd)
        
        # Retornar la ruta del archivo de salida
        return str(self.out_path)

    # Método para separar el audio usando el modelo de Demucs
    def un_audio(self, file):
        
        print(f"Going to separate the file: {file}")
        
        # Construir el comando para Demucs como una lista de argumentos
        cmd = [
            "-o", str(self.out_path), 
            "-n", self.model
            
        ]
        
        # Eliminar la opción MP3 si no se necesita
        if self.mp3:
            cmd += ["--mp3", f"--mp3-bitrate={self.mp3_rate}"]

        if self.float32:
            cmd += ["--float32"]
        if self.int24:
            cmd += ["--int24"]

        # Incluir el archivo de audio
        cmd.append(str(file))

        # Ejecutar demucs.separate.main() pasando los argumentos
        #print(f"With command: {cmd}")
        out = demucs.separate.mainv2(cmd)
        #out = demucs.separate.main(cmd)
        #print(out)
        return out

        
    
    # Método para obtener los resultados sin guardarlos como wav
    def obtener_resultados(self):
        # Llamar a la separación
        result_path = self.separate()
        
        if result_path:
            # Puedes procesar los archivos resultantes de alguna forma si es necesario
            separated_files = list(Path(result_path).iterdir())
            return separated_files
        return []

    # Método para guardar el resultado del audio en la carpeta de salida
    def guardar_audio(self):
        result_path = self.separate()
        if result_path:
            print(f"Archivos separados guardados en: {result_path}")
            print("Puedes ver los archivos separados en la carpeta de salida.")
        else:
            print("No se pudo realizar la separación.")






def prevent_clip(wav, mode='rescale'):
    """
    different strategies for avoiding raw clipping.
    """
    if mode is None or mode == 'none':
        return wav
    assert wav.dtype.is_floating_point, "too late for clipping"
    if mode == 'rescale':
        wav = wav / max(1.01 * wav.abs().max(), 1)
    elif mode == 'clamp':
        wav = wav.clamp(-0.99, 0.99)
    elif mode == 'tanh':
        wav = torch.tanh(wav)
    else:
        raise ValueError(f"Invalid mode {mode}")
    return wav


def save_audio(wav: torch.Tensor,
               path: tp.Union[str, Path],
               samplerate: int,
               clip: tp.Literal["rescale", "clamp", "tanh", "none"] = 'rescale',
               bits_per_sample: tp.Literal[16, 24, 32] = 16):
    """Save audio file, automatically preventing clipping if necessary
    based on the given `clip` strategy. If the path ends in `.mp3`, this
    will save as mp3 with the given `bitrate`. Use `preset` to set mp3 quality:
    2 for highest quality, 7 for fastest speed
    """
    wav = prevent_clip(wav, mode=clip)
    path = Path(path)
    suffix = path.suffix.lower()
    
    bits_per_sample = 32
    
    encoding = 'PCM_S'
    
    ta.save(str(path), wav, sample_rate=samplerate,
                encoding=encoding, bits_per_sample=bits_per_sample)
    print("guardado...")


from demucs.pretrained import get_model, ModelLoadingError
from demucs.separate import load_track
from demucs.apply import apply_model
from dora.log import fatal
import torchaudio


def cargar_demucs():
    model = get_model('htdemucs')
    try:
        model = get_model('htdemucs')
    except ModelLoadingError as error:
        fatal(error.args[0])
    model.cpu()
    model.eval()
    return model
        

def separar_demucs(audio_path=None, model= None):
    if audio_path != None and model != None:
        wav = load_track(audio_path, model.audio_channels, model.samplerate)

        ref = wav.mean(0)
        wav -= ref.mean()
        wav /= ref.std()
        sources = apply_model(model, wav[None], device='cuda', shifts=1,
                                split=True, overlap=0.25, progress=True,
                                num_workers=4, segment=None)[0]
        sources *= ref.std()
        sources += ref.mean()

        lista_estereo = list(sources)
        i= 1
        #Primera muestra es drums
        #Segundo es bass
        #Tercero es others
        #Cuarto es vocales

        # Crear una lista vacía para almacenar los waveforms convertidos a mono
        lista_mono = []

        # Convertir cada waveform estéreo en la lista a mono
        for waveform in lista_estereo:
            # Convertir de estéreo a mono (promediando los canales)
            mono_waveform = torch.mean(waveform, dim=0, keepdim=True)
            lista_mono.append(mono_waveform)


        #for data in lista_mono:
        #    print(data.shape)
        #    torchaudio.save(f'output_stereo{i}.wav', data, 44100)
        #    i+=1
        return lista_mono
    else:
        print("Debes introducir una direccion de audio...")
        return None

def cargar_audio(audio_path=None, audio_channels=2, sample_rate=44100):
    if audio_path != None:
        wav = load_track(audio_path, audio_channels, sample_rate)
    else:
        raise ValueError(f"Error debes introducir una direccion correcta.")

    return wav



def separar_eval_demucs(wav=None, model= None):
    if wav != None and model != None:
        ref = wav.mean(0)
        wav -= ref.mean()
        wav /= ref.std()
        sources = apply_model(model, wav[None], device='cuda', shifts=1,
                                split=True, overlap=0.25, progress=True,
                                num_workers=4, segment=None)[0]
        sources *= ref.std()
        sources += ref.mean()

        lista_estereo = list(sources)
        i= 1
        #Primera muestra es drums
        #Segundo es bass
        #Tercero es others
        #Cuarto es vocales

        # Crear una lista vacía para almacenar los waveforms convertidos a mono
        lista_mono = []


        #for data in lista_mono:
        #    print(data.shape)
        #    torchaudio.save(f'output_stereo{i}.wav', data, 44100)
        #    i+=1
        return lista_estereo
    else:
        print("Debes introducir una direccion de audio...")
        return None



def separar_demucs_mono(wav=None, model= None):
    if wav != None and model != None:
        ref = wav.mean(0)
        wav -= ref.mean()
        wav /= ref.std()
        sources = apply_model(model, wav[None], device='cuda', shifts=1,
                                split=True, overlap=0.25, progress=True,
                                num_workers=4, segment=None)[0]
        sources *= ref.std()
        sources += ref.mean()

        lista_estereo = list(sources)
        i= 1
        #Primera muestra es drums
        #Segundo es bass
        #Tercero es others
        #Cuarto es vocales

        # Crear una lista vacía para almacenar los waveforms convertidos a mono
        lista_mono = []
        for audio in lista_estereo:
            lista_mono.append(convertir_a_mono(audio))
            
        return lista_mono
    else:
        print("Debes introducir una direccion de audio...")
        return None
