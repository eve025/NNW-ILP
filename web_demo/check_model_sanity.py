# web_demo/check_model_sanity.py
import os, pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

APP_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(APP_DIR, '..', 'lstm_notebook', 'spam_lstm_model.h5')
TOKENIZER_PATH = os.path.join(APP_DIR, '..', 'lstm_notebook', 'tokenizer.pkl')
MAX_SEQ_LEN = 100

assert os.path.exists(MODEL_PATH) and os.path.exists(TOKENIZER_PATH), "Modelo o tokenizer faltan."

model = load_model(MODEL_PATH)
with open(TOKENIZER_PATH, 'rb') as f:
    tokenizer = pickle.load(f)

print('Modelo cargado:', MODEL_PATH)
print('Vocab size:', len(getattr(tokenizer,'word_index',{})))

# Inspect last Dense layer weights/bias
last_dense = None
for layer in reversed(model.layers):
    if hasattr(layer, 'get_weights'):
        w = layer.get_weights()
        if w and w[0].ndim == 2 and w[1].ndim == 1 and w[1].shape[0] == 1:
            last_dense = layer
            break

if last_dense is None:
    print('No pude detectar la última capa Dense (con bias escalar). Mostrar capas:')
    for i,layer in enumerate(model.layers):
        print(i, layer.name, type(layer))
else:
    W, b = last_dense.get_weights()
    print('Última Dense encontrada:', last_dense.name)
    print('W shape:', W.shape)
    print('W mean:', W.mean(), 'std:', W.std())
    print('b shape:', b.shape, 'b[0]:', float(b[0]))

# Prepare zero input and random input
zero_input = np.zeros((1, MAX_SEQ_LEN), dtype='int32')
rand_input = np.random.randint(1, max(2, len(getattr(tokenizer,'word_index',{}))+1), size=(1, MAX_SEQ_LEN))

p_zero = float(model.predict(zero_input)[0][0])
p_rand = float(model.predict(rand_input)[0][0])
print('\nPredicción entrada todo-cero:', p_zero)
print('Predicción entrada aleatoria (tokens válidos posibles):', p_rand)

# Also predict on the example set used antes
examples = [
    "FELICIDADES! Has ganado $1000. Reclama aquí: http://bit.ly/ganar",
    "Hola, ¿nos vemos mañana a las 10?",
    "URGENTE: Actualiza tus datos bancarios para evitar bloqueo.",
    "Oferta especial, compra ahora con 90% de descuento!!!"
]
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import unicodedata, re

def clean_text(text):
    text = str(text).lower()
    text = unicodedata.normalize('NFKD', text).encode('ascii','ignore').decode('ascii')
    text = re.sub(r'http\S+|www\S+', ' ', text)
    text = re.sub(r'\S+@\S+', ' ', text)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

cleaned = [clean_text(t) for t in examples]
sequences = [tokenizer.texts_to_sequences([c])[0] for c in cleaned]
padded = pad_sequences(sequences, maxlen=MAX_SEQ_LEN, padding='post')
print('\nEjemplos padded nonzero counts:')
for i in range(padded.shape[0]):
    print(i, int((padded[i]>0).sum()), sequences[i][:30])

preds = model.predict(padded)
print('Preds ejemplos:', preds.reshape(-1))

print('\nSi `predicción entrada todo-cero` ≈ la prob que viste antes (0.48467...), entonces el modelo está devolviendo el bias como salida y las entradas no cambian la salida de forma significativa.')
print('Terminado.')
