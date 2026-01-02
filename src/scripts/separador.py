import torch
import torchaudio
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.unet import UNet
from src.models.disc import Discriminator
from src.models.demucsm import SepararAudio, separar_demucs, cargar_demucs, cargar_audio, separar_demucs_mono


from src.data.preprocesar import mel_spectrogram
from src.data.preprocesar import dividir_en_segmentos
from src.utils.spectrograms import normalizar_espectrograma_mel, split_spectrogram, nivelar_mel
from src.utils.waveforms import unir_wav, nivelar_waveforms, guardar_waveform
from src.utils.datos import plot_mel_spectrogram

# Función para cargar el modelo entrenado
def cargar_modelo(path_modelo: str, device: torch.device = None):
    modelo = UNet().to(device)
    checkpoint = torch.load(path_modelo, map_location=device)
    modelo.load_state_dict(checkpoint['generator_state_dict'])
    modelo.eval()  # Coloca el modelo en modo evaluación
    return modelo


def cargar_vocoder(device: torch.device = None):
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


def procesar_vocoder(data, model, corte):
    data = split_spectrogram(data)
    ls_temp = []
    with torch.inference_mode():
        i = 0
        for mel in tqdm(data, leave=False):
            mel = mel.to(device='cuda')
            wav_gen = model(mel)
            ls_temp.extend(list(wav_gen))
            if i == 10 and corte:
                break
            i+=1
    salida = unir_wav(ls_temp)
    return salida

import torchaudio
class Refinador:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.unet_d = cargar_modelo(R"data\drums_entrenmiento\checkpoint_epoch_50.pth", self.device)
        self.unet_b = cargar_modelo(R"data\bass_entrenmiento\checkpoint_epoch_50.pth", self.device)
        self.vocoder = cargar_vocoder(self.device)
        self.demucs = cargar_demucs()

    def cargar_audio(self, path):
        wav = cargar_audio(path)

    def get_mel(self, wav):
        out = separar_demucs_mono(wav, self.demucs)
        drums = out[0]
        bass = out[1]
        others = out[2]
        vocals = out[3]

        #Espectrogramas mel
        drums_mel = mel_spectrogram(drums)
        bass_mel = mel_spectrogram(bass)
        others_mel = mel_spectrogram(others)
        vocals_mel = mel_spectrogram(vocals)
        drums_mel = procesar_unet(drums_mel, self.unet_d)
        bass_mel = procesar_unet(bass_mel, self.unet_b)
        
        ls_instrumentos =  [drums_mel, bass_mel, others_mel, vocals_mel]
        return nivelar_mel(ls_instrumentos)


    def procesar(self, wav, corte = True):
        out = separar_demucs_mono(wav, self.demucs)
        drums = out[0]
        bass = out[1]
        others = out[2]
        vocals = out[3]

        #Espectrogramas mel
        drums_mel = mel_spectrogram(drums)
        bass_mel = mel_spectrogram(bass)
        others_mel = mel_spectrogram(others)
        vocals_mel = mel_spectrogram(vocals)
        
        data = procesar_unet(drums_mel, self.unet_d)
        drums = procesar_vocoder(data, self.vocoder, corte)
        
        data = procesar_unet(bass_mel, self.unet_b)
        bass = procesar_vocoder(data, self.vocoder, corte)

        ls_instrumentos =  [drums, bass, others, vocals]

        return nivelar_waveforms(ls_instrumentos)


if __name__ == '__main__':
    modelo = Refinador()

    wav = cargar_audio(R"C:\Users\Luis\Documents\DATASETS\pruebasss\mixture.wav")
    out = modelo.procesar(wav, corte=False)
    nombres = ['drums.wav', 'bass.wav', 'others.wav', 'vocals.wav']
    out_path = R'C:\Users\Luis\Downloads\pruebas'
    for i, nombre in enumerate(nombres):
        guardar_waveform(out[i], nombre, 44100, out_path)


