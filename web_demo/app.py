from flask import Flask, render_template, request, jsonify
import os
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from flask import send_from_directory

APP_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(APP_DIR, '..', 'lstm_notebook', 'spam_lstm_model.h5')
TOKENIZER_PATH = os.path.join(APP_DIR, '..', 'lstm_notebook', 'tokenizer.pkl')
MAX_SEQ_LEN = 100
# Umbral sugerido para marcar spam (se puede ajustar tras evaluar el modelo)
# El valor puede sobrescribirse leyendo lstm_notebook/best_threshold.txt
THRESHOLD = 0.3
BEST_THRESHOLD_PATH = os.path.join(APP_DIR, '..', 'lstm_notebook', 'best_threshold.txt')

app = Flask(__name__)

# Cargar modelo y tokenizer al iniciar (si existen)
model = None
tokenizer = None
VOCAB_LIMIT = None

if os.path.exists(MODEL_PATH) and os.path.exists(TOKENIZER_PATH):
    model = load_model(MODEL_PATH)
    with open(TOKENIZER_PATH, 'rb') as f:
        tokenizer = pickle.load(f)
    # intentar detectar la dimensión del vocabulario que espera la capa Embedding
    try:
        emb = next((l for l in model.layers if l.__class__.__name__.lower() == 'embedding'), None)
        if emb is not None and hasattr(emb, 'input_dim'):
            VOCAB_LIMIT = int(emb.input_dim)
            print('Detected embedding input_dim (vocab limit):', VOCAB_LIMIT)
    except Exception:
        VOCAB_LIMIT = None
else:
    print('Aviso: modelo o tokenizer no encontrados. Ejecuta el entrenamiento primero y coloca los archivos en ../lstm_notebook/')

# Intentar leer el mejor umbral calculado en el entrenamiento
if os.path.exists(BEST_THRESHOLD_PATH):
    try:
        with open(BEST_THRESHOLD_PATH, 'r', encoding='utf-8') as bf:
            v = bf.read().strip()
            if v:
                THRESHOLD = float(v)
                print(f'Umbral cargado desde {BEST_THRESHOLD_PATH}: {THRESHOLD}')
    except Exception as e:
        print(f'No se pudo leer {BEST_THRESHOLD_PATH}: {e}. Usando THRESHOLD={THRESHOLD}')


def clean_text(text):
    import re
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', ' ', text)
    text = re.sub(r'\S+@\S+', ' ', text)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/report')
def report():
    # sirve training_report.txt si existe
    report_file = os.path.join(os.path.dirname(MODEL_PATH), 'training_report.txt')
    if os.path.exists(report_file):
        return send_from_directory(os.path.dirname(report_file), os.path.basename(report_file))
    return 'Reporte no encontrado. Entrena el modelo primero.', 404


@app.route('/plot')
def plot():
    plot_file = os.path.join(os.path.dirname(MODEL_PATH), 'training_plot.png')
    if os.path.exists(plot_file):
        return send_from_directory(os.path.dirname(plot_file), os.path.basename(plot_file))
    return 'Gráfica no encontrada. Entrena el modelo primero.', 404

@app.route('/predict', methods=['POST'])
def predict():
    global model, tokenizer
    if model is None or tokenizer is None:
        return jsonify({'error': 'Modelo no cargado. Entrena y copia los archivos.'}), 500
    text = request.form.get('message', '')
    clean = clean_text(text)
    seq = tokenizer.texts_to_sequences([clean])[0]
    # proteger contra índices mayores que los que la capa Embedding admite
    if VOCAB_LIMIT is not None:
        seq = [int(i) if (isinstance(i, int) and i < VOCAB_LIMIT) else 1 for i in seq]
    seq_p = pad_sequences([seq], maxlen=MAX_SEQ_LEN, padding='post')
    # recargar umbral en cada petición para reflejar re-entrenamientos sin reiniciar el servidor
    try:
        if os.path.exists(BEST_THRESHOLD_PATH):
            with open(BEST_THRESHOLD_PATH, 'r', encoding='utf-8') as bf:
                v = bf.read().strip()
                if v:
                    # actualizar THRESHOLD local para esta petición
                    threshold_used = float(v)
                else:
                    threshold_used = THRESHOLD
        else:
            threshold_used = THRESHOLD
    except Exception:
        threshold_used = THRESHOLD
    pred = float(model.predict(seq_p)[0][0])
    label_en = 'spam' if pred > threshold_used else 'ham'
    # Traducción/etiqueta en español (más legible)
    label_es = 'Spam' if label_en == 'spam' else 'Legítimo'
    display = label_es
    # Añadir información útil para la UI: umbral y mensaje de asesoramiento
    advice = ''
    if pred < 0.15:
        advice = 'Confianza muy baja — el modelo no está seguro; considera reentrenar con más datos.'
    elif pred < THRESHOLD and pred > 0.15:
        advice = 'Confianza baja — puedes ajustar el umbral o reentrenar para mejorar detección.'
    return jsonify({
        'prediction_en': label_en,
        'prediction_es': label_es,
        'display': display,
        'score': pred,
        'threshold': threshold_used,
        'advice': advice
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
