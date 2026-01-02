import os
import re
import shutil

def numerar_carpetas(direccion_principal, sufijo="all_", empieza=151):
    """
    Renombra todas las carpetas dentro de la carpeta principal, asignando un sufijo con un número incremental.

    :param direccion_principal: Dirección de la carpeta principal que contiene las subcarpetas.
    :param sufijo: Sufijo base para el nuevo nombre de las carpetas, seguido por un número incremental.
    """
    try:
        # Verificar si la dirección es válida
        if not os.path.isdir(direccion_principal):
            print(f"La dirección {direccion_principal} no es válida o no es una carpeta.")
            return
        
        # Listar todas las carpetas en la dirección principal
        carpetas = [carpeta for carpeta in os.listdir(direccion_principal)
                    if os.path.isdir(os.path.join(direccion_principal, carpeta))]
        
        # Renombrar las carpetas con el sufijo y un número incremental
        for i, carpeta in enumerate(carpetas, start=empieza):
            carpeta_path = os.path.join(direccion_principal, carpeta)
            nuevo_nombre = f"{sufijo}{i}"
            nuevo_path = os.path.join(direccion_principal, nuevo_nombre)
            
            # Renombrar la carpeta
            os.rename(carpeta_path, nuevo_path)
            print(f"Carpeta renombrada: {carpeta} -> {nuevo_nombre}")
    
    except Exception as e:
        print(f"Ocurrió un error: {e}")


def mover_archivos_wav(carpeta_principal):
    # Verifica si la carpeta principal existe
    if not os.path.isdir(carpeta_principal):
        print(f"La carpeta {carpeta_principal} no existe.")
        return

    # Recorre todas las subcarpetas dentro de la carpeta principal
    for root, dirs, files in os.walk(carpeta_principal):
        # Evita mover archivos en la carpeta principal misma
        if root == carpeta_principal:
            continue
        
        for file in files:
            # Si el archivo es un .wav
            if file.lower().endswith(".wav"):
                # Obtiene la ruta completa del archivo
                archivo_origen = os.path.join(root, file)
                archivo_destino = os.path.join(carpeta_principal, file)
                
                # Si ya existe un archivo con el mismo nombre, cambia el nombre del archivo para evitar sobrescribirlo
                if os.path.exists(archivo_destino):
                    base, extension = os.path.splitext(file)
                    contador = 1
                    nuevo_nombre = f"{base}_{contador}{extension}"
                    archivo_destino = os.path.join(carpeta_principal, nuevo_nombre)
                    while os.path.exists(archivo_destino):
                        contador += 1
                        nuevo_nombre = f"{base}_{contador}{extension}"
                        archivo_destino = os.path.join(carpeta_principal, nuevo_nombre)
                
                # Mueve el archivo a la carpeta principal
                shutil.move(archivo_origen, archivo_destino)
                print(f"Archivo {file} movido a {archivo_destino}")




def recolectar_archivos(carpeta):
    # Lista para almacenar las rutas completas de los archivos que cumplen con el patrón
    archivos_validos = []
    
    # Expresión regular para encontrar archivos con el patrón 'all_n.wav'
    patron = re.compile(r'^all_\d+\.wav$')

    # Recorrer todos los archivos en la carpeta
    for archivo in os.listdir(carpeta):
        ruta_completa = os.path.join(carpeta, archivo)
        if patron.match(archivo) and os.path.isfile(ruta_completa):  # Verificamos si es un archivo
            archivos_validos.append(ruta_completa)

    return archivos_validos


def renombrar_archivos_en_carpetas(carpeta_principal):
    # Recorremos todas las carpetas dentro de la carpeta principal
    for carpeta in os.listdir(carpeta_principal):
        carpeta_path = os.path.join(carpeta_principal, carpeta)
        
        # Verificamos si es una carpeta
        if os.path.isdir(carpeta_path):
            # Recorremos los archivos dentro de la carpeta
            for archivo in os.listdir(carpeta_path):
                archivo_path = os.path.join(carpeta_path, archivo)
                
                # Verificamos si el archivo es un archivo .wav
                if archivo.endswith('.wav') and os.path.isfile(archivo_path):
                    # Creamos el nuevo nombre para el archivo
                    nuevo_nombre = f"{carpeta}_{archivo}"
                    nuevo_path = os.path.join(carpeta_path, nuevo_nombre)
                    
                    # Renombramos el archivo
                    os.rename(archivo_path, nuevo_path)
                    print(f"Renombrado: {archivo} -> {nuevo_nombre}")

                    

def renombrar_archivos(carpeta, instrumento, nuevo_nombre):
    # Obtiene una lista de todos los archivos en la carpeta
    archivos = os.listdir(carpeta)
    
    # Filtra solo los archivos wav que contienen el nombre del instrumento
    for archivo in archivos:
        if archivo.endswith('.wav'):
            tmp1 = os.path.splitext(archivo)[0]
            # Extrae el prefijo y el sufijo del archivo
            partes = tmp1.split('_')
            if len(partes) == 3 and instrumento in partes:  # Esto asegura que el archivo tiene la forma 'all_XX_instrument.wav'
                numero = partes[1]  # Obtiene el número (ejemplo: 01, 02, 03)
                # Renombra el archivo
                if nuevo_nombre != "":
                    nuevo_nombre_archivo = f"all_{numero}_{nuevo_nombre}.wav"
                else:
                    nuevo_nombre_archivo = f"all_{numero}.wav"

                # Construye las rutas completas de los archivos
                ruta_original = os.path.join(carpeta, archivo)
                ruta_nueva = os.path.join(carpeta, nuevo_nombre_archivo)
                # Renombra el archivo
                os.rename(ruta_original, ruta_nueva)
                print(f"Renombrado: {archivo} -> {nuevo_nombre_archivo}")




if __name__ =="__main__":
    #ls_archivos = recolectar_archivos(R"C:\Users\Luis\Documents\DATASETS\DSD100\target")
    #carpeta_salida = R"C:\Users\Luis\Documents\DATASETS\DSD100\input"
    #
    #model = demucs.SepararAudio(out_path=carpeta_salida)
    #
    #for data in ls_archivos:
    #    model.un_audio(data)
    #renombrar_archivos_en_carpetas(R"C:\Users\Luis\Documents\DATASETS\DSD100\input\htdemucs")
    #renombrar_archivos(R"C:\Users\Luis\Documents\DATASETS\musdb18hq\train\input\htdemucs", "mixture", "")

    mover_archivos_wav(R"C:\Users\Luis\Documents\DATASETS\DSD100\input\htdemucs")
#
    #model = demucs.SepararAudio(out_path=carpeta_salida)
    #model.un_audio(R"C:\Users\Luis\Documents\DATASETS\musdb18hq\train\target\all_11.wav")

