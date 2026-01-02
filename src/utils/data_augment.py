import torch
import torch.nn.functional as F
import random





def quitar_frecuencias(tensor_imagenes, min_ancho, max_ancho, cantidad_lineas, min_fila=0):
    """
    Agrega líneas horizontales negras de ancho aleatorio a las imágenes en un tensor.
    Las líneas son generadas de manera independiente para cada imagen del batch.

    Parameters:
    - tensor_imagenes (Tensor): Tensor de imágenes con forma [batch_size, 1, altura, anchura].
    - min_ancho (int): Ancho mínimo de las líneas en píxeles.
    - max_ancho (int): Ancho máximo de las líneas en píxeles.
    - cantidad_lineas (int): Número de líneas horizontales a agregar.
    - min_fila (int): Fila mínima (desde la cual se pueden agregar las líneas horizontales).

    Returns:
    - Tensor modificado con las líneas horizontales negras.
    """
    # Asegurarse de que el tensor de imágenes es del tamaño correcto
    batch_size, _, altura, anchura = tensor_imagenes.shape

    # Asegurar que la fila mínima esté dentro del rango válido
    min_fila = max(min_fila, 0)  # No permitir que min_fila sea negativa
    max_fila = altura - 1  # Última fila disponible

    # Iterar sobre cada imagen del batch
    for i in range(batch_size):
        for _ in range(cantidad_lineas):
            # Elegir aleatoriamente una fila de inicio para la línea, asegurándonos que esté en el rango válido
            fila_inicio = random.randint(min_fila, max_fila)

            # Elegir aleatoriamente un ancho entre el mínimo y máximo
            ancho = random.randint(min_ancho, max_ancho)

            # Asegurarse de que el final no se salga del rango de la imagen
            fila_fin = min(fila_inicio + ancho, max_fila)

            # Establecer a 0 (negro) las filas seleccionadas en la imagen i
            tensor_imagenes[i, :, fila_inicio:fila_fin + 1, :] = 0

    return tensor_imagenes




def modificar_freq(input_tensor, target_tensor, bandas_a_cortar):
    """
    Deforma dos tensores de la forma [32, 1, 128, 128], cortando aleatoriamente el número de bandas especificado
    desde las bandas superiores (frecuencias altas) y luego interpolando para devolverlos a la forma [32, 1, 128, 128].
    Los cortes deben aplicarse de forma coherente a ambos tensores para que estén alineados.
    
    input_tensor: Tensor de entrada con forma [32, 1, 128, 128].
    target_tensor: Tensor de objetivo con forma [32, 1, 128, 128].
    bandas_a_cortar: Número de bandas que se deben cortar (por ejemplo, 20 bandas).
    
    Retorna un par de tensores deformados con la forma [32, 1, 128, 128].
    """
    
    # Listas para almacenar los tensores deformados
    input_deformado = []
    target_deformado = []
    
    for i in range(input_tensor.shape[0]):
        # Realizar el corte para el espectrograma i
        corte = random.randint(0, bandas_a_cortar)
        
        # Cortar la parte superior (frecuencias altas) del espectrograma
        # Corte desde el índice corte hacia la parte superior
        input_cortado = input_tensor[i:i+1, :, :128-corte, :]
        target_cortado = target_tensor[i:i+1, :, :128-corte, :]
        
        # Interpolar para devolver el tamaño original [1, 1, 128, 128]
        input_interpolado = F.interpolate(input_cortado, size=(128, 128), mode='bilinear', align_corners=False)
        target_interpolado = F.interpolate(target_cortado, size=(128, 128), mode='bilinear', align_corners=False)
        
        # Añadir los tensores deformados a sus respectivas listas
        input_deformado.append(input_interpolado)
        target_deformado.append(target_interpolado)
    
    # Convertir las listas de tensores deformados en un tensor final de forma [32, 1, 128, 128]
    input_deformado = torch.cat(input_deformado, dim=0)
    target_deformado = torch.cat(target_deformado, dim=0)
    
    return input_deformado, target_deformado

