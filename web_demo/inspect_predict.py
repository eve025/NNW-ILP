# web_demo/inspect_predict.py
import os, pickle
import numpy as np
import re, unicodedata
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

APP_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(APP_DIR, '..', 'lstm_notebook', 'spam_lstm_model.h5')
TOKENIZER_PATH = os.path.join(APP_DIR, '..', 'lstm_notebook', 'tokenizer.pkl')
MAX_SEQ_LEN = 100

def clean_text(text):
    # Usa la misma normalización que en quick_test (quita acentos y caracteres no alfanuméricos)
    text = str(text).lower()
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii')
    text = re.sub(r'http\\S+|www\\S+', ' ', text)
    text = re.sub(r'\\S+@\\S+', ' ', text)
    text = re.sub(r'[^a-z0-9\\s]', ' ', text)
    text = re.sub(r'\\s+', ' ', text).strip()
    return text

assert os.path.exists(MODEL_PATH), "Modelo no encontrado: " + MODEL_PATH
assert os.path.exists(TOKENIZER_PATH), "Tokenizer no encontrado: " + TOKENIZER_PATH

print("Cargando modelo:", MODEL_PATH)
model = load_model(MODEL_PATH)
print("Cargando tokenizer:", TOKENIZER_PATH)
with open(TOKENIZER_PATH, 'rb') as f:
    tokenizer = pickle.load(f)

# Información del tokenizer
wi = getattr(tokenizer, 'word_index', None)
num_words = getattr(tokenizer, 'num_words', None)
print("Tokenizador: word_index size =", len(wi) if wi is not None else 'None')
print("Tokenizador.num_words =", num_words)
# Muestra 20 entradas del vocab (si existen)
if wi:
    sample_items = list(wi.items())[:20]
    print("Primeras 20 entradas del vocab (word -> idx):")
    for w, idx in sample_items:
        print(f"  {w} -> {idx}")

examples = [
    "FELICIDADES! Has ganado $1000. Reclama aquí: http://bit.ly/ganar",
    "Hola, ¿nos vemos mañana a las 10?",
    "URGENTE: Actualiza tus datos bancarios para evitar bloqueo.",
    "Oferta especial, compra ahora con 90% de descuento!!!"
]

for t in examples:
    clean = clean_text(t)
    seq = tokenizer.texts_to_sequences([clean])
    padded = pad_sequences(seq, maxlen=MAX_SEQ_LEN, padding='post')
    print("\nTexto original:", t)
    print("Texto limpio  :", clean)
    print("Secuencia     :", seq)
    print("Padded sum    :", int(padded.sum()), " nonzero count:", int((padded>0).sum()))
    pred = float(model.predict(padded)[0][0])
    print("Predicción prob(spam) =", pred)

# Predicción para vector cero (todas las posiciones 0)
zero = np.zeros((1, MAX_SEQ_LEN), dtype=np.int32)
print("\nPredicción para vector cero (all zeros):", float(model.predict(zero)[0][0]))

# Comparar con una secuencia aleatoria de índices válidos (si hay vocab)
if wi:
    # crear secuencia que contenga algunos índices del vocab (ej: los 5 primeros índices que existan)
    some_idxs = [idx for _, idx in list(wi.items())[:5]]
    test_seq = np.zeros((1, MAX_SEQ_LEN), dtype=np.int32)
    for i, idx in enumerate(some_idxs):
        if i < MAX_SEQ_LEN:
            test_seq[0, i] = idx
    print("Predicción para secuencia construida con primeros indices del vocab:", float(model.predict(test_seq)[0][0]))
else:
    print("No hay word_index para construir secuencia de prueba.")