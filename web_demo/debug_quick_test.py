# web_demo/debug_quick_test.py
import os, pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import unicodedata, re

APP_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(APP_DIR, '..', 'lstm_notebook', 'spam_lstm_model.h5')
TOKENIZER_PATH = os.path.join(APP_DIR, '..', 'lstm_notebook', 'tokenizer.pkl')
BEST_THRESH = os.path.join(APP_DIR, '..', 'lstm_notebook', 'best_threshold.txt')
MAX_SEQ_LEN = 100

def clean_text(text):
    text = str(text).lower()
    text = unicodedata.normalize('NFKD', text).encode('ascii','ignore').decode('ascii')
    text = re.sub(r'http\\S+|www\\S+', ' ', text)
    text = re.sub(r'\\S+@\\S+', ' ', text)
    text = re.sub(r'[^a-z0-9\\s]', ' ', text)
    text = re.sub(r'\\s+', ' ', text).strip()
    return text

assert os.path.exists(MODEL_PATH) and os.path.exists(TOKENIZER_PATH), "Modelo o tokenizer faltan."

model = load_model(MODEL_PATH)
with open(TOKENIZER_PATH, 'rb') as f:
    tokenizer = pickle.load(f)

try:
    thresh = float(open(BEST_THRESH,'r',encoding='utf-8').read().strip())
except Exception:
    thresh = 0.5

print("Usando umbral:", thresh)
print("Vocab size:", len(getattr(tokenizer,'word_index',{})))
print()

examples = [
    "FELICIDADES! Has ganado $1000. Reclama aquí: http://bit.ly/ganar",
    "Hola, ¿nos vemos mañana a las 10?",
    "URGENTE: Actualiza tus datos bancarios para evitar bloqueo.",
    "Oferta especial, compra ahora con 90% de descuento!!!"
]

for t in examples:
    c = clean_text(t)
    seq = tokenizer.texts_to_sequences([c])[0]
    padded = pad_sequences([seq], maxlen=MAX_SEQ_LEN, padding='post')
    pred = float(model.predict(padded)[0][0])
    print("ORIGINAL:", t)
    print("CLEAN:", c)
    print("SEQUENCE (first 30):", seq[:30])
    print("NONZERO count (padded):", int((padded>0).sum()))
    print("RAW prob:", pred)
    print("LABEL(@0.5):", "Spam" if pred>0.5 else "Legítimo")
    print("LABEL(@best):", "Spam" if pred>thresh else "Legítimo")
    print("-"*60)