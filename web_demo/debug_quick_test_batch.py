# web_demo/debug_quick_test_batch.py
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
    text = re.sub(r'http\S+|www\S+', ' ', text)
    text = re.sub(r'\S+@\S+', ' ', text)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
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

cleaned = [clean_text(t) for t in examples]
sequences = [tokenizer.texts_to_sequences([c])[0] for c in cleaned]
for i, (orig, c, seq) in enumerate(zip(examples, cleaned, sequences)):
    print(f"--- Example {i} ---")
    print("ORIGINAL:", orig)
    print("CLEAN:", c)
    print("SEQUENCE (first 50):", seq[:50])
    print("SEQUENCE length:", len(seq))

padded = pad_sequences(sequences, maxlen=MAX_SEQ_LEN, padding='post')
print('\nPadded shape:', padded.shape)
print('Padded (first 3 rows, show up to 50 cols):')
for i in range(padded.shape[0]):
    print(f'row {i} nonzero_count:', int((padded[i]>0).sum()))
    print(padded[i,:50])

print('\nRunning batch predict...')
preds = model.predict(padded, batch_size=len(padded))
print('Preds shape:', preds.shape)
print('Preds (flat):', preds.reshape(-1))

# Also predict each separately to compare
print('\nRunning per-sample predict...')
indiv = [float(model.predict(np.expand_dims(padded[i],0))[0][0]) for i in range(padded.shape[0])]
print('Individual preds:', indiv)

print('\nCompare differences:')
for i in range(len(indiv)):
    print(f'Example {i}: seq_len={len(sequences[i])} nonzeros={(padded[i]>0).sum()} pred_batch={preds[i,0]:.6f} pred_indiv={indiv[i]:.6f}')

print('\nLabels (@0.5 and @best):')
for i,p in enumerate(preds.reshape(-1)):
    print(f'Example {i}: RAW={p:.6f} LABEL@0.5={"Spam" if p>0.5 else "Legit"} LABEL@best={"Spam" if p>thresh else "Legit"}')

print('\nTokenizer sample mappings (first 20 words):')
wi = getattr(tokenizer, 'word_index', {})
items = list(wi.items())[:20]
for w, idx in items:
    print(w, idx)

print('\nDone.')
