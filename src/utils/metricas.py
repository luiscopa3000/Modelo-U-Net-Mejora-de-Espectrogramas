import torch
import torch.nn.functional as F
import numpy as np
import librosa
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM

from src.data.preprocesar import mel_spectrogram
from src.utils.datos import plot_mel_spectrogram
from src.data.preprocesar import dividir_en_segmentos


def SDR_calcular(ref_img: torch.Tensor, est_img: torch.Tensor, epsilon=1e-8):
    """
    Calcula el SDR para imágenes/espectrogramas sin consumir RAM excesiva
    Forma esperada: [batch, height, width] o [height, width]
    """
    # Asegurar que los tensores estén en CPU y sean float32
    ref = ref_img.detach().cpu().float()
    est = est_img.detach().cpu().float()
    
    # Calcular error
    error = ref - est
    
    # Calcular potencias
    signal_power = torch.sum(ref**2)
    noise_power = torch.sum(error**2)
    
    # Evitar división por cero
    sdr = (10 * torch.log10((signal_power + epsilon) / (noise_power + epsilon)))
    
    return sdr.item()


def calculate_ssim(img1: torch.Tensor, img2: torch.Tensor):
    """
    Versión para tensores PyTorch con forma [1, H, W]
    """
    # Añadir dimensión de canales (NCHW)
    img1 = img1.unsqueeze(1)  # [1, 1, H, W]
    img2 = img2.unsqueeze(1)
    
    # Calcular SSIM
    ssim = SSIM(data_range=1.0).to(img1.device)
    return ssim(img1, img2).item()


def calcular_mse(imagen_original, imagen_predicha):
    # Calcular y devolver el MSE
    mse = F.mse_loss(imagen_predicha, imagen_original)
    return mse.item()


def calcular_media(inputs, target):
    norm_input = (inputs-inputs.min())/(inputs.max()-inputs.min())
    norm_target = (target-target.min())/(target.max()-target.min())
    ls_inputs = dividir_en_segmentos(norm_input)
    ls_target = dividir_en_segmentos(norm_target)
    sdr_t = 0
    ssim_t = 0
    mse_t = 0
    for i in range(len(ls_target)):
        mel_inputs = ls_inputs[i]
        mel_target = ls_target[i]
        sdr_t += SDR_calcular(mel_target, mel_inputs)
        ssim_t += calculate_ssim(mel_target, mel_inputs)
        mse_t += calcular_mse(mel_target, mel_inputs)
    sdr_t /= len(ls_target)
    ssim_t /= len(ls_target)
    mse_t /= len(ls_target)
    return sdr_t, ssim_t.item(), mse_t



def SSIM_media(inputs, target):
    norm_input = (inputs-inputs.min())/(inputs.max()-inputs.min())
    norm_target = (target-target.min())/(target.max()-target.min())
    ls_inputs = dividir_en_segmentos(norm_input)
    ls_target = dividir_en_segmentos(norm_target)
    ssim_t = 0
    for i in range(len(ls_target)):
        mel_inputs = ls_inputs[i]
        mel_target = ls_target[i]
        ssim_t += calculate_ssim(mel_target, mel_inputs)
    ssim_t /= len(ls_target)
    return ssim_t.item()




#inputs_wav, sr = librosa.load(R"C:\Users\Luis\Documents\DATASETS\musdb18hq\test\input\all_01_bass.wav", sr=44100, mono=True) # wav is np.ndarray with shape [T_time] and values in [-1, 1]
#inputs_wav = torch.FloatTensor(inputs_wav).unsqueeze(0) # wav is FloatTensor with shape [B(1), T_time]
#
#
#target_wav, sr = librosa.load(R"C:\Users\Luis\Documents\DATASETS\musdb18hq\test\target\all_01_bass.wav", sr=44100, mono=True) # wav is np.ndarray with shape [T_time] and values in [-1, 1]
#target_wav = torch.FloatTensor(target_wav).unsqueeze(0) # wav is FloatTensor with shape [B(1), T_time]
#
#
#bass, sr = librosa.load(R"bass.wav", sr=44100, mono=True) # wav is np.ndarray with shape [T_time] and values in [-1, 1]
#bass = torch.FloatTensor(bass).unsqueeze(0) # wav is FloatTensor with shape [B(1), T_time]
#
#
#
#input_mel = mel_spectrogram(inputs_wav)
#target_mel = mel_spectrogram(target_wav)
#bass_mel = mel_spectrogram(bass)
#
#out = calcular_media(input_mel, target_mel)
#print(out)
#print(bass_mel.shape)


# Ejemplo de uso
#bass_input = bass_mel
#original = input_mel[:,:,:1408] # [batch, H, W]
#predicted = target_mel[:,:,:1408]
#
#print(f"SDR optimizado: {SDR_calcular(original, predicted):.2f} dB")
#print(f"SDR optimizado: {SDR_calcular(original, bass_input):.2f} dB")
#
#ssim_bass = ((bass_mel-bass_mel.min())/(bass_mel.max()-bass_mel.min()))[:,:,:128]
#ssim_input = ((input_mel-input_mel.min())/(input_mel.max()-input_mel.min()))[:,:,:128]
#ssim_target = ((target_mel-target_mel.min())/(target_mel.max()-target_mel.min()))[:,:,:128]
#
#plot_mel_spectrogram(ssim_bass)
#plot_mel_spectrogram(ssim_input)
#plot_mel_spectrogram(ssim_target)
#print(f"SSIM (GPU): {calculate_ssim_torch(ssim_target, ssim_input):.4f}")
#print(f"SSIM (GPU): {calculate_ssim_torch(ssim_target, ssim_bass):.4f}")





#from torchmetrics.audio import SignalDistortionRatio
#import torch
#
#
#target_wav = target_wav[:,:720896]
#inputs_wav = inputs_wav[:,:720896]
#sdr = SignalDistortionRatio()
#out = sdr(inputs_wav, target_wav)
#out = sdr(bass, target_wav)
#
#
#
#print(bass.shape)
#print(out)

#Calcular SDR completo
