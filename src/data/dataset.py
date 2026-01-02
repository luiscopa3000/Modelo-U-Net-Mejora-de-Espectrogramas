import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd

class Audio_dataset(Dataset):
    """
    Dataset para cargar espectrogramas mel segmentados desde archivos .pt generados previamente.
    """
    def __init__(self, csv_file, transform=None, normalize = True):
        """
        Args:
            csv_file (str): Path al archivo CSV con la información de segmentación.
            transform (callable, optional): Transformaciones opcionales que se aplican a los tensores.
        """
        # Cargar el CSV con la información de los segmentos.
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.normalize = normalize

    def __len__(self):
        # Número de segmentos en el CSV.
        return len(self.data)

    def __getitem__(self, idx):
        # Obtener información del segmento.
        tensor_file = self.data.iloc[idx]['ruta']

        # Cargar el tensor .pt del archivo correspondiente.
        data = torch.load(tensor_file)
        input_tensor = data['input']  # Espectrograma de entrada (input).
        target_tensor = data['target']  # Espectrograma de salida (target)
        
        if self.normalize:
            # Normalización al rango [0, 1]
            if input_tensor.max() != input_tensor.min():
                input_tensor = (input_tensor - input_tensor.min()) / (input_tensor.max() - input_tensor.min())
            else:
                input_tensor = (input_tensor - input_tensor.min()) / (input_tensor.max() - input_tensor.min()+1e-6)

            if target_tensor.max() != target_tensor.min():
                target_tensor = (target_tensor - target_tensor.min()) / (target_tensor.max() - target_tensor.min())
            else: 
                target_tensor = (target_tensor - target_tensor.min()) / (target_tensor.max() - target_tensor.min()+1e-6)

        # Si hay transformaciones, aplicarlas
        if self.transform:
            input_tensor = self.transform(input_tensor)
            target_tensor = self.transform(target_tensor)
        
        return input_tensor, target_tensor









