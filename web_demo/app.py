from flask import Flask, render_template, request, jsonify
import os
import sys
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from flask import send_from_directory

# Make project root importable so `from lstm_notebook...` works when running from web_demo/
APP_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(APP_DIR, '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Custom objects mapping to handle models saved with custom ops/layers
# Add entries as needed if other custom layers/functions are present.
CUSTOM_OBJECTS = {
    'NotEqual': tf.math.not_equal,
    'Equal': tf.math.equal,
}
# Prefer refit artifacts when available (generated during reentrenos)
MODEL_PATH = os.path.join(APP_DIR, '..', 'lstm_notebook', 'spam_lstm_model_refit.h5')
ALT_MODEL_PATH = os.path.join(APP_DIR, '..', 'lstm_notebook', 'spam_lstm_model.h5')
TOKENIZER_PATH = os.path.join(APP_DIR, '..', 'lstm_notebook', 'tokenizer_refit.pkl')
ALT_TOKENIZER_PATH = os.path.join(APP_DIR, '..', 'lstm_notebook', 'tokenizer.pkl')
MAX_SEQ_LEN = 100
# Umbral sugerido para marcar spam (se puede ajustar tras evaluar el modelo)
# El valor puede sobrescribirse leyendo lstm_notebook/best_threshold.txt
THRESHOLD = 0.3
BEST_THRESHOLD_PATH = os.path.join(APP_DIR, '..', 'lstm_notebook', 'best_threshold_refit.txt')
ALT_BEST_THRESHOLD_PATH = os.path.join(APP_DIR, '..', 'lstm_notebook', 'best_threshold.txt')

app = Flask(__name__)

# Cargar modelo y tokenizer al iniciar (si existen)
model = None
tokenizer = None
VOCAB_LIMIT = None

def try_load_model_and_tokenizer():
    global model, tokenizer, VOCAB_LIMIT, MAX_SEQ_LEN
    # pick available artifacts (prefer refit)
    model_file = MODEL_PATH if os.path.exists(MODEL_PATH) else (ALT_MODEL_PATH if os.path.exists(ALT_MODEL_PATH) else None)
    tok_file = TOKENIZER_PATH if os.path.exists(TOKENIZER_PATH) else (ALT_TOKENIZER_PATH if os.path.exists(ALT_TOKENIZER_PATH) else None)
    if model_file is None or tok_file is None:
        return False
    # load tokenizer
    try:
        with open(tok_file, 'rb') as f:
            tokenizer_local = pickle.load(f)
    except Exception as e:
        print('Error cargando tokenizer:', e)
        return False

    # try to load full model (first attempting with custom_object_scope to handle custom ops)
    try:
        try:
            with tf.keras.utils.custom_object_scope(CUSTOM_OBJECTS):
                model_local = load_model(model_file)
        except Exception:
            # fallback: try loading without scope (some models don't need it)
            model_local = load_model(model_file)
    except Exception as e:
        print('Warning: load_model failed (custom scope attempted). Attempting to rebuild architecture and load weights:', e)
        try:
            # attempt to import build_model from retrain script
            from lstm_notebook.retrain_with_hard_negatives import build_model
        except Exception as e2:
            print('No se pudo importar build_model():', e2)
            return False
        # locate weights snapshot next to model file or known fallback
        weights_path = None
        candidate = os.path.join(os.path.dirname(model_file), os.path.splitext(os.path.basename(model_file))[0] + '.weights.h5')
        if os.path.exists(candidate):
            weights_path = candidate
        else:
            altw = os.path.join(os.path.dirname(model_file), 'spam_lstm_model_refit.weights.h5')
            if os.path.exists(altw):
                weights_path = altw
        if weights_path is None:
            print('Weights snapshot not found next to model file; cannot rebuild architecture')
            return False
        model_local = build_model()
        model_local.load_weights(weights_path)
        print('Loaded model architecture and weights from', weights_path)

    model = model_local
    tokenizer = tokenizer_local
    # intentar inferir MAX_SEQ_LEN desde la forma de entrada del modelo (si está disponible)
    try:
        inp_shape = getattr(model, 'input_shape', None)
        if inp_shape is not None:
            if isinstance(inp_shape, tuple) and len(inp_shape) > 1 and inp_shape[1] is not None:
                MAX_SEQ_LEN = int(inp_shape[1])
                print('Detected MAX_SEQ_LEN from model.input_shape:', MAX_SEQ_LEN)
            elif isinstance(inp_shape, (list, tuple)) and len(inp_shape) > 0:
                first = inp_shape[0]
                if isinstance(first, tuple) and len(first) > 1 and first[1] is not None:
                    MAX_SEQ_LEN = int(first[1])
                    print('Detected MAX_SEQ_LEN from model.input_shape list:', MAX_SEQ_LEN)
    except Exception:
        pass
    # intentar detectar la dimensión del vocabulario que espera la capa Embedding
    try:
        emb = next((l for l in model.layers if l.__class__.__name__.lower() == 'embedding'), None)
        if emb is not None and hasattr(emb, 'input_dim'):
            VOCAB_LIMIT = int(emb.input_dim)
            print('Detected embedding input_dim (vocab limit):', VOCAB_LIMIT)
    except Exception:
        pass
    return True


if not try_load_model_and_tokenizer():
    print('Aviso: modelo o tokenizer no encontrados o falló la carga. Ejecuta el entrenamiento y coloca los archivos en ../lstm_notebook/')

# Intentar leer el mejor umbral calculado en el entrenamiento
best_path_used = BEST_THRESHOLD_PATH if os.path.exists(BEST_THRESHOLD_PATH) else (ALT_BEST_THRESHOLD_PATH if os.path.exists(ALT_BEST_THRESHOLD_PATH) else None)
if best_path_used is not None:
    try:
        with open(best_path_used, 'r', encoding='utf-8') as bf:
            v = bf.read().strip()
            if v:
                THRESHOLD = float(v)
                print(f'Umbral cargado desde {best_path_used}: {THRESHOLD}')
    except Exception as e:
        print(f'No se pudo leer {best_path_used}: {e}. Usando THRESHOLD={THRESHOLD}')


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
    # Añadir información útil para la UI: umbral (sin mensajes de "modelo no está seguro")
    # Eliminamos el mensaje explícito de falta de seguridad del modelo en la respuesta.
    response = {
        'prediction_en': label_en,
        'prediction_es': label_es,
        'display': display,
        'score': pred,
        'threshold': threshold_used
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
