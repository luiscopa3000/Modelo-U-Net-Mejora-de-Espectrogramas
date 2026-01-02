import torch

def normalizar_espectrograma_mel(espectrograma):
    """
    Normaliza un espectrograma mel para que sus valores estén en el rango [0, 1].
    
    Args:
    espectrograma (torch.Tensor): El espectrograma mel (tensor 2D o 3D).
    
    Returns:
    torch.Tensor: Espectrograma normalizado en el rango [0, 1].
    """
    # Encontrar el valor mínimo y máximo del espectrograma
    min_val = espectrograma.min()
    max_val = espectrograma.max()
    
    # Normalizar el espectrograma al rango [0, 1]
    espectrograma_normalizado = (espectrograma - min_val) / (max_val - min_val)
    
    return espectrograma_normalizado


def split_spectrogram(spec: torch.Tensor, window_size: int = 128, overlap_frames: int = 20):
    """
    Divide un espectrograma de forma [1,128,n_frames] en una lista de espectrogramas [1,128,window_size],
    con un solapamiento fijo en número de frames.

    Parámetros:
        spec (torch.Tensor): Tensor con forma [1,128,n_frames]
        window_size (int): Tamaño de ventana (default 128)
        overlap_frames (int): Número de frames que se solapan entre ventanas consecutivas

    Retorna:
        List[torch.Tensor]: Lista de tensores con forma [1,128,window_size]
    """
    assert 0 <= overlap_frames < window_size, "El solapamiento debe ser menor que el tamaño de ventana"

    stride = window_size - overlap_frames
    n_frames = spec.shape[2]

    spec_windows = []
    for start in range(0, n_frames - window_size + 1, stride):
        end = start + window_size
        window = spec[:, :, start:end]
        spec_windows.append(window)

    return spec_windows



def nivelar_mel(tensores):
    """
    Recibe una lista de tensores con waveforms de diferentes longitudes y los recorta para que todos
    tengan la misma longitud temporal (n_frames), que será igual al tensor con menos frames.

    Args:
    tensores (list of torch.Tensor): Lista de tensores con forma [1, 128, n_frames].

    Returns:
    list of torch.Tensor: Lista de tensores recortados al tamaño del tensor con menos frames.
    """
    
    # Encuentra la longitud mínima (n_frames más corto)
    min_frames = min(tensor.shape[2] for tensor in tensores)
    
    # Recorta todos los tensores a la misma longitud temporal
    tensores_recortados = [tensor[:, :, :min_frames] for tensor in tensores]
    
    return tensores_recortados


