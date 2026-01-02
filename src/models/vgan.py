device = 'cuda'

import torch
from src.models.big_v_gan import bigvgan
import librosa
from src.models.big_v_gan.meldataset import get_mel_spectrogram
import torchaudio

import torch
class Vocoder:
    def __init__(self, model_name='nvidia/bigvgan_v2_44khz_128band_512x', device=None):
        """
        Inicializa el vocoder, carga el modelo BigVGAN y lo mueve al dispositivo adecuado.
        
        Args:
            model_name (str): Nombre o ruta del modelo preentrenado de BigVGAN.
            device (torch.device, optional): El dispositivo donde se ejecutará el modelo (por defecto None).
        """
        # Establecer el dispositivo
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Cargar el modelo BigVGAN
        self.model = bigvgan.BigVGAN.from_pretrained(model_name, use_cuda_kernel=False)
        self.model.remove_weight_norm()
        self.model = self.model.eval().to(self.device)
        
    def procesar(self, mel):
        """
        Genera la forma de onda a partir de un espectrograma Mel utilizando BigVGAN.
        
        Args:
            mel (torch.Tensor): Espectrograma Mel de entrada (debe ser de forma [B, 1, T_time]).
        
        Returns:
            torch.Tensor: Forma de onda generada como un tensor de tipo Float (de forma [B, 1, T_time]).
        """
        # Generar la forma de onda a partir del espectrograma Mel
        with torch.inference_mode():
            wav_gen = self.model(mel)  # wav_gen es un FloatTensor con forma [B, 1, T_time]
        
        return wav_gen
    










#model = bigvgan.BigVGAN.from_pretrained('nvidia/bigvgan_v2_44khz_128band_512x', use_cuda_kernel=False)
#
#model.remove_weight_norm()
#model = model.eval().to(device)
#
## cargar archivo wav y calcular el espectrograma mel
#wav_path = R"C:\Users\Luis\Documents\DATASETS\musdb18hq\test\input\all_01_bass.wav"
#wav, sr = librosa.load(wav_path, sr=model.h.sampling_rate, mono=True) # wav es un np.ndarray con forma [T_time] y valores en [-1, 1]
#wav = torch.FloatTensor(wav).unsqueeze(0) # wav es un FloatTensor con forma [B(1), T_time]
#
## calcular el espectrograma mel a partir del audio de referencia
#mel = get_mel_spectrogram(wav, model.h).to(device) # mel es un FloatTensor con forma [B(1), C_mel, T_frame]
#mel = mel[:,:,128:1000]
#
#print(mel.shape)
#
#
#
#
## generar la forma de onda a partir del mel
#with torch.inference_mode():
#    wav_gen = model(mel) # wav_gen es un FloatTensor con forma [B(1), 1, T_time] y valores en [-1, 1]
#wav_gen_float = wav_gen.squeeze(0).cpu() # wav_gen es un FloatTensor con forma [1, T_time]
#print(wav_gen_float.shape)
## puedes convertir la forma de onda generada a PCM lineal de 16 bits
#wav_gen_int16 = (wav_gen_float * 32767.0).numpy().astype('int16') # wav_gen ahora es un np.ndarray con forma [1, T_time] y tipo de dato int16
#
#
## Guardar el tensor float directamente (sin convertir a int16)
#torchaudio.save(
#    "audio_generado_test.wav",
#    wav_gen_float,  # Tensor float32 con forma [1, T] (valores en [-1, 1])
#    sample_rate=44100,  # 44000 Hz
#    encoding="PCM_S",  # Formato estándar para WAV
#    bits_per_sample=16,  # 16 bits (calidad estándar)
#)

