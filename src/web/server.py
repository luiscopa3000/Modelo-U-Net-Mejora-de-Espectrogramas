from flask import Flask, request, send_file, render_template, session, jsonify
import os
import uuid
import threading
import time
from src.models.demucsm import cargar_demucs, separar_demucs, save_audio
from src.web.service import procesar, procesa
import torch
from flasgger import Swagger

from pydub import AudioSegment

app = Flask(__name__)
app.secret_key = '534523123'
lock = threading.Lock()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

swagger_config = {
    "headers": [],
    "specs": [
        {
            "endpoint": 'apispec_1',
            "route": '/apispec_1.json',
            "rule_filter": lambda rule: True,  # incluir todas las rutas
            "model_filter": lambda tag: True,  # incluir todos los modelos
        }
    ],
    "static_url_path": "/flasgger_static",
    "swagger_ui": True,
    "specs_route": "/apidocs/"
}

swagger = Swagger(app, config=swagger_config)

#CARGAR MODELO
m_dem = cargar_demucs()

# Configuración de directorios
UPLOAD_FOLDER = 'data/temp/temp_uploads'
OUTPUT_FOLDER = R'C:\Users\Luis\Desktop\final_proy\env\src\data\temp\temp_outputs'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


def convert_to_wav(input_file_path):
    """
    Convierte un archivo de audio a formato WAV usando pydub.
    
    Parámetros:
        input_file_path (str): Ruta al archivo de entrada.
    
    Retorna:
        str: Ruta al archivo WAV resultante.
    """
    # Verificamos si el archivo ya es un .wav
    if input_file_path.lower().endswith('.wav'):
        #print("El archivo ya es un WAV. No es necesario convertirlo.")
        return input_file_path  # Devolvemos el archivo original sin hacer cambios
    
    # Si no es WAV, procedemos a convertirlo
    #print(f"Convirtiendo el archivo {input_file_path} a WAV...")
    audio = AudioSegment.from_file(input_file_path)
    output_file_path = input_file_path.rsplit('.', 1)[0] + '.wav'  # Reemplaza la extensión con .wav
    audio.export(output_file_path, format='wav')
    return output_file_path

# Ruta principal con interfaz web
@app.route('/')
def index():
    if 'user_id' not in session:
        session['user_id'] = uuid.uuid4().hex
    return render_template('index.html', user_id=session['user_id'])


# Ruta para separación de audio
@app.route('/subir', methods=['POST'])
def separate_audio():
    """
    Subida y separación de audio en stems.
    ---
    consumes:
      - multipart/form-data
    parameters:
      - name: audio
        in: formData
        type: file
        required: true
        description: Archivo de audio a procesar
    responses:
      200:
        description: Audio procesado exitosamente
        schema:
          type: object
          properties:
            success:
              type: string
              example: Archivo audio.wav cargado y procesando en segundo plano.
      400:
        description: Error en los datos enviados
      500:
        description: Error interno al procesar el audio
    """
    if 'audio' not in request.files:
        return {'error': 'No audio file provided'}, 400
    
    file = request.files['audio']
    if file.filename == '':
        return {'error': 'Empty filename'}, 400

    input_path = os.path.join(UPLOAD_FOLDER, f"{session['user_id']}_{file.filename}")
    try:
        file.save(input_path)
        
        # Convertir cualquier archivo de audio a .wav
        input_path = convert_to_wav(input_path)

        with lock:
            procesar(input_path, m_dem, OUTPUT_FOLDER, session['user_id'])
            #procesa(input_path, OUTPUT_FOLDER, session['user_id'])

        #return {'download_url': f'/download/{session['user_id']}', 'filename': file.filename}
        return jsonify({'success': f'Archivo {file.filename} cargado y procesando en segundo plano.'}), 200
        
            
    except Exception as e:
        return {'error': str(e)}, 500
    
    finally:
        if os.path.exists(input_path):
            os.remove(input_path)


# Ruta para servir los archivos procesados
@app.route('/download/<filename>')
def download_file(filename):
    """
    Descarga el archivo de audio procesado.
    ---
    parameters:
      - name: filename
        in: path
        type: string
        required: true
        description: Nombre del archivo a descargar
    responses:
      200:
        description: Archivo enviado correctamente
      404:
        description: Archivo no encontrado
    """
    try:
        # Usar send_file para enviar el archivo al cliente
        file_path = os.path.join(OUTPUT_FOLDER, filename)
        #print(OUTPUT_FOLDER, file_path)
        return send_file(file_path, as_attachment=False)  # as_attachment=False para mostrar el archivo en el navegador
    except FileNotFoundError:
        return jsonify({'error': 'Archivo no encontrado'}), 404


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)















