import os
import torch
import torchaudio


def convertir_a_mono(tensor):
    """
    Convierte un tensor estéreo (forma [2, n_frames]) a mono (forma [n_frames]).

    Args:
    tensor (torch.Tensor): Tensor de forma [2, n_frames], representando un audio estéreo.

    Returns:
    torch.Tensor: Tensor de forma [n_frames], representando el audio en mono.
    """
    # Verifica que el tensor tenga la forma esperada
    if tensor.shape[0] != 2:
        raise ValueError("El tensor debe tener 2 canales (forma [2, n_frames])")

    # Promediamos los dos canales
    tensor_mono = tensor.mean(dim=0, keepdim=True)  # Promedio a lo largo de la dimensión 0 (canales)

    return tensor_mono


def crossfade_waveforms(w1: torch.Tensor, w2: torch.Tensor, overlap: int) -> torch.Tensor:
    """
    Une dos waveforms con una zona de solapamiento usando crossfade con ventana Hann.

    Parámetros:
        w1 (torch.Tensor): Primer waveform, forma [1, N]
        w2 (torch.Tensor): Segundo waveform, forma [1, M]
        overlap (int): Tamaño de la zona solapada

    Retorna:
        torch.Tensor: Waveform unido con crossfade
    """
    assert w1.shape[0] == 1 and w2.shape[0] == 1, "Se espera forma [1, N]"
    assert w1.shape[1] >= overlap and w2.shape[1] >= overlap, "Los waveforms deben tener suficiente solapamiento"

    # Zonas relevantes
    non_overlap_1 = w1[:, :-overlap]
    overlap_1 = w1[:, -overlap:]

    overlap_2 = w2[:, :overlap]
    non_overlap_2 = w2[:, overlap:]

    # Ventanas de crossfade
    fade_out = torch.linspace(1, 0, steps=overlap).to(w1.device)
    fade_in = torch.linspace(0, 1, steps=overlap).to(w1.device)

    # Aplicar ventanas
    crossfaded = (overlap_1 * fade_out) + (overlap_2 * fade_in)
    crossfaded = crossfaded.unsqueeze(0) if crossfaded.ndim == 1 else crossfaded

    # Unir todo
    joined = torch.cat([non_overlap_1, crossfaded, non_overlap_2], dim=1)
    return joined

def unir_wav(ls_wavs):
    wav = ls_wavs[0]
    for i in range(1,len(ls_wavs)):
        wav = crossfade_waveforms(wav, ls_wavs[i], 10240)
    return wav



def nivelar_waveforms(tensores):
    """
    Recibe una lista de tensores con waveforms de diferentes longitudes y los recorta para que todos
    tengan la misma longitud, la cual será igual al tensor más corto de la lista.

    Args:
    tensores (list of torch.Tensor): Lista de tensores con forma [2, n_frames], cada uno representando un waveform.

    Returns:
    list of torch.Tensor: Lista de tensores recortados al tamaño del tensor con menos frames.
    """
    
    # Encuentra la longitud del tensor con menos frames
    min_frames = min(tensor.shape[1] for tensor in tensores)
    
    # Recorta todos los tensores para que tengan la misma longitud
    tensores_recortados = [tensor[:, :min_frames] for tensor in tensores]
    
    return tensores_recortados

def guardar_waveform(waveform, nombre, sample_rate, ruta_guardado):
    """
    Guarda un tensor de waveform como un archivo WAV en la ruta especificada.

    Args:
    waveform (torch.Tensor): Tensor de forma [1, n_frames], representando el waveform.
    nombre (str): Nombre del archivo WAV (por ejemplo, 'audio.wav').
    sample_rate (int): Frecuencia de muestreo del audio (por ejemplo, 44100).
    ruta_guardado (str): Ruta del directorio donde se guardará el archivo WAV.
    """
    # Verifica si el tensor está en la GPU, y si es así, muévelo a la CPU
    if waveform.is_cuda:
        print("El waveform está en la GPU, moviéndolo a la CPU...")
        waveform = waveform.cpu()

    # Asegúrate de que la ruta de guardado exista
    if not os.path.exists(ruta_guardado):
        os.makedirs(ruta_guardado)  # Crea el directorio si no existe
    
    # Construye la ruta completa para guardar el archivo
    ruta_completa = os.path.join(ruta_guardado, nombre)
    
    # Guarda el archivo en la ruta especificada
    torchaudio.save(ruta_completa, waveform, sample_rate)
    
    print(f"Archivo guardado en: {ruta_completa}")