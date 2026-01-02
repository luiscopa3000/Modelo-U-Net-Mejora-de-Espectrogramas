
import os
import torch
import csv

def save_metrics_to_csv(epoch, g_loss_epoch, d_loss_epoch, avg_adv_loss, avg_l1_loss, 
                        g_val_loss_epoch, d_val_loss_epoch, val_avg_adv, val_avg_l1, 
                        filename='training_metrics.csv'):
    """
    Guarda las métricas de entrenamiento y validación en un archivo CSV.

    Args:
        epoch (int): La época actual.
        g_loss_epoch (float): Pérdida del generador en entrenamiento (promediada).
        d_loss_epoch (float): Pérdida del discriminador en entrenamiento (promediada).
        avg_adv_loss (float): Pérdida adversarial promedio en entrenamiento.
        avg_l1_loss (float): Pérdida L1 promedio en entrenamiento.
        g_val_loss_epoch (float): Pérdida del generador en validación (promediada).
        d_val_loss_epoch (float): Pérdida del discriminador en validación (promediada).
        val_avg_adv (float): Pérdida adversarial promedio en validación.
        val_avg_l1 (float): Pérdida L1 promedio en validación.
        filename (str): Nombre del archivo CSV donde se guardarán los datos.
    """
    # Verificamos si el archivo ya existe para decidir si escribir encabezados o no
    file_exists = False
    try:
        with open(filename, 'r') as file:
            file_exists = True
    except FileNotFoundError:
        file_exists = False

    # Si el archivo no existe, escribimos los encabezados
    if not file_exists:
        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            # Escribimos el encabezado
            writer.writerow([
                'Epoch', 'G_loss_train', 'D_loss_train', 'Avg_adv_loss_train', 'Avg_L1_loss_train',
                'G_loss_val', 'D_loss_val', 'Avg_adv_loss_val', 'Avg_L1_loss_val'
            ])
    
    # Ahora escribimos los datos de la época
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            epoch, g_loss_epoch, d_loss_epoch, avg_adv_loss, avg_l1_loss, 
            g_val_loss_epoch, d_val_loss_epoch, val_avg_adv, val_avg_l1
        ])



def save_checkpoint(generator, discriminator, opt_g, opt_d,
                    epoch, g_loss_epoch, d_loss_epoch, 
                    avg_adv_loss, avg_l1_loss, g_val_loss_epoch, d_val_loss_epoch, 
                    val_avg_adv, val_avg_l1, checkpoint_path):
    """
    Guarda el estado de los modelos, optimizadores, scheduler y métricas a un archivo checkpoint.

    Args:
        generator (nn.Module): El modelo generador.
        discriminator (nn.Module): El modelo discriminador.
        opt_g (torch.optim.Optimizer): El optimizador del generador.
        opt_d (torch.optim.Optimizer): El optimizador del discriminador.
        scheduler_g (torch.optim.lr_scheduler): Scheduler del generador.
        epoch (int): La época actual.
        g_loss_epoch (float): Pérdida del generador en entrenamiento (promediada).
        d_loss_epoch (float): Pérdida del discriminador en entrenamiento (promediada).
        avg_adv_loss (float): Pérdida adversarial promedio en entrenamiento.
        avg_l1_loss (float): Pérdida L1 promedio en entrenamiento.
        g_val_loss_epoch (float): Pérdida del generador en validación (promediada).
        d_val_loss_epoch (float): Pérdida del discriminador en validación (promediada).
        val_avg_adv (float): Pérdida adversarial promedio en validación.
        val_avg_l1 (float): Pérdida L1 promedio en validación.
        checkpoint_path (str): Ruta para guardar el archivo checkpoint.
    """
    # Verificar si el directorio existe y crearlo si no existe
    checkpoint_dir = os.path.dirname(checkpoint_path)  # Obtiene el directorio de la ruta
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)  # Crear directorio si no existe

    checkpoint = {
        'epoch': epoch,
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'optimizer_g_state_dict': opt_g.state_dict(),
        'optimizer_d_state_dict': opt_d.state_dict(),
        'g_loss_epoch': g_loss_epoch,
        'd_loss_epoch': d_loss_epoch,
        'avg_adv_loss': avg_adv_loss,
        'avg_l1_loss': avg_l1_loss,
        'g_val_loss_epoch': g_val_loss_epoch,
        'd_val_loss_epoch': d_val_loss_epoch,
        'val_avg_adv': val_avg_adv,
        'val_avg_l1': val_avg_l1,
    }

    torch.save(checkpoint, checkpoint_path)
    # print(f"Checkpoint guardado en {checkpoint_path} para la época {epoch}.")


def load_checkpoint(generator, discriminator, opt_g, opt_d, checkpoint_path, device):
    """
    Carga el estado de los modelos, optimizadores, scheduler y métricas desde un archivo checkpoint.

    Args:
        generator (nn.Module): El modelo generador.
        discriminator (nn.Module): El modelo discriminador.
        opt_g (torch.optim.Optimizer): El optimizador del generador.
        opt_d (torch.optim.Optimizer): El optimizador del discriminador.
        scheduler_g (torch.optim.lr_scheduler): Scheduler del generador.
        checkpoint_path (str): Ruta del archivo checkpoint.
        device (torch.device): El dispositivo (CPU o GPU) para cargar el modelo.
    
    Returns:
        epoch (int): La última época entrenada.
        g_loss_epoch (float): Última pérdida del generador en entrenamiento.
        d_loss_epoch (float): Última pérdida del discriminador en entrenamiento.
        avg_adv_loss (float): Última pérdida adversarial promedio en entrenamiento.
        avg_l1_loss (float): Última pérdida L1 promedio en entrenamiento.
        g_val_loss_epoch (float): Última pérdida del generador en validación.
        d_val_loss_epoch (float): Última pérdida del discriminador en validación.
        val_avg_adv (float): Última pérdida adversarial promedio en validación.
        val_avg_l1 (float): Última pérdida L1 promedio en validación.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Cargar el estado de los modelos
    generator.load_state_dict(checkpoint['generator_state_dict'])
    discriminator.load_state_dict(checkpoint['discriminator_state_dict'])

    # Cargar los optimizadores
    opt_g.load_state_dict(checkpoint['optimizer_g_state_dict'])
    opt_d.load_state_dict(checkpoint['optimizer_d_state_dict'])


    # Recuperar las métricas
    epoch = checkpoint['epoch']
    g_loss_epoch = checkpoint['g_loss_epoch']
    d_loss_epoch = checkpoint['d_loss_epoch']
    avg_adv_loss = checkpoint['avg_adv_loss']
    avg_l1_loss = checkpoint['avg_l1_loss']
    g_val_loss_epoch = checkpoint['g_val_loss_epoch']
    d_val_loss_epoch = checkpoint['d_val_loss_epoch']
    val_avg_adv = checkpoint['val_avg_adv']
    val_avg_l1 = checkpoint['val_avg_l1']

    print(f"Checkpoint cargado desde {checkpoint_path}. Reanudando desde la época {epoch}.")

    return epoch, g_loss_epoch, d_loss_epoch, avg_adv_loss, avg_l1_loss, g_val_loss_epoch, d_val_loss_epoch, val_avg_adv, val_avg_l1







