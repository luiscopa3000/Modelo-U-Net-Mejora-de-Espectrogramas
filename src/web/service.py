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
from src.models.demucsm import cargar_demucs, separar_demucs, save_audio, cargar_audio

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from src.scripts.separador import Refinador


# Función para cargar el modelo entrenado
def mod_peso(modelo, path_modelo: str, device: torch.device = None):
    checkpoint = torch.load(path_modelo, map_location=device)
    modelo.load_state_dict(checkpoint['generator_state_dict'])
    return modelo


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



def procesar(all, demucs, OUTPUT_FOLDER, unique_id):
    print(all)
    out = separar_demucs(audio_path=all, model=demucs)
    d_drums = out[0] #Drums 
    d_bass = out[1] #Bass
    d_others = out[2] #Others
    d_vocals = out[3] #Vocals

    save_audio(
        d_drums,
        os.path.join(OUTPUT_FOLDER, f"drums_{unique_id}.wav"),
        samplerate=44100
    )
    
    save_audio(
        d_bass,
        os.path.join(OUTPUT_FOLDER, f"bass_{unique_id}.wav"),
        samplerate=44100
    )
    
    save_audio(
        d_vocals,
        os.path.join(OUTPUT_FOLDER, f"vocals_{unique_id}.wav"),
        samplerate=44100
    )

    save_audio(
        d_others,
        os.path.join(OUTPUT_FOLDER, f"others_{unique_id}.wav"),
        samplerate=44100
    )


def procesa(all, OUTPUT_FOLDER, unique_id):
    modelo = Refinador()
    
    wav = cargar_audio(all)
    out = modelo.procesar(wav, corte=False)

    d_drums = out[0].cpu() #Drums 
    d_bass = out[1].cpu() #Bass
    d_others = out[2].cpu() #Others
    d_vocals = out[3].cpu() #Vocals

    save_audio(
        d_drums,
        os.path.join(OUTPUT_FOLDER, f"drums_{unique_id}.wav"),
        samplerate=44100
    )
    
    save_audio(
        d_bass,
        os.path.join(OUTPUT_FOLDER, f"bass_{unique_id}.wav"),
        samplerate=44100
    )
    
    save_audio(
        d_vocals,
        os.path.join(OUTPUT_FOLDER, f"vocals_{unique_id}.wav"),
        samplerate=44100
    )

    save_audio(
        d_others,
        os.path.join(OUTPUT_FOLDER, f"others_{unique_id}.wav"),
        samplerate=44100
    )

























