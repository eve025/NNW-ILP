# web_demo/quick_test.py
import os, pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

APP_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(APP_DIR, '..', 'lstm_notebook', 'spam_lstm_model.h5')
TOKENIZER_PATH = os.path.join(APP_DIR, '..', 'lstm_notebook', 'tokenizer.pkl')
MAX_SEQ_LEN = 100

def clean_text(text):
    import re, unicodedata
    text = str(text).lower()
    # quitar acentos
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

examples = [
    "FELICIDADES! Has ganado $1000. Reclama aquí: http://bit.ly/ganar",
    "Hola, ¿nos vemos mañana a las 10?",
    "URGENTE: Actualiza tus datos bancarios para evitar bloqueo.",
    "Oferta especial, compra ahora con 90% de descuento!!!"
]

for t in examples:
    clean = clean_text(t)
    seq = tokenizer.texts_to_sequences([clean])
    seq_p = pad_sequences(seq, maxlen=MAX_SEQ_LEN, padding='post')
    pred = float(model.predict(seq_p)[0][0])
    print(f"Texto: {t}")
    print(f"-> Prob(spam) = {pred:.4f}  Etiqueta actual (umbral 0.5): {'Spam' if pred>0.5 else 'Legítimo'}")
    print("-"*60)