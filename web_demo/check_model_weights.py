import os
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

APP_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(APP_DIR, '..', 'lstm_notebook', 'spam_lstm_model.h5')
TOKENIZER_PATH = os.path.join(APP_DIR, '..', 'lstm_notebook', 'tokenizer.pkl')
MAX_SEQ_LEN = 100

def load_resources():
    assert os.path.exists(MODEL_PATH), f"Modelo no encontrado: {MODEL_PATH}"
    assert os.path.exists(TOKENIZER_PATH), f"Tokenizer no encontrado: {TOKENIZER_PATH}"
    print('Cargando modelo...')
    model = load_model(MODEL_PATH)
    print('Cargando tokenizer...')
    with open(TOKENIZER_PATH, 'rb') as f:
        tokenizer = pickle.load(f)
    return model, tokenizer

def inspect_last_layer(model):
    # asumir que la última capa es Dense(1)
    last = model.layers[-1]
    print('\nÚltima capa:', last)
    try:
        w, b = last.get_weights()
        print('Weights shape:', w.shape)
        print('Bias shape:', b.shape)
        print('Weights stats: mean={:.6f}, std={:.6f}, min={:.6f}, max={:.6f}'.format(
            np.mean(w), np.std(w), np.min(w), np.max(w)))
        print('Bias stats: mean={:.6f}, std={:.6f}, min={:.6f}, max={:.6f}'.format(
            np.mean(b), np.std(b), np.min(b), np.max(b)))
    except Exception as e:
        print('No se pudieron leer pesos de la última capa:', e)

def try_examples(model, tokenizer):
    examples = [
        'free entry win prize now',
        'hello how are you doing today',
        'claim your prize http now',
    ]
    print('\nTokenizador word_index size =', len(getattr(tokenizer, 'word_index', {})))
    for t in examples:
        seq = tokenizer.texts_to_sequences([t])
        padded = pad_sequences(seq, maxlen=MAX_SEQ_LEN, padding='post')
        print('\nEjemplo:', t)
        print('  secuencia (primeros 20):', seq[0][:20])
        print('  nonzero count:', int((padded>0).sum()))
        pred = float(model.predict(padded)[0][0])
        print('  pred prob(spam)=', pred)

    # prueba con vectores construidos: zeros, índices frecuentes, índices al azar
    zero = np.zeros((1, MAX_SEQ_LEN), dtype=np.int32)
    print('\nPredicción para vector cero (all zeros):', float(model.predict(zero)[0][0]))

    # construir secuencia con índices frecuentes si existen
    wi = getattr(tokenizer, 'word_index', {})
    if wi:
        top_words = list(wi.items())[:10]
        idxs = [idx for _, idx in top_words if idx < 1000]
        if idxs:
            seq = np.zeros((1, MAX_SEQ_LEN), dtype=np.int32)
            for i, idx in enumerate(idxs[:MAX_SEQ_LEN]):
                seq[0, i] = idx
            print('Predicción para secuencia construida con índices frecuentes:', float(model.predict(seq)[0][0]))

    # secuencia con un token de 'free' si existe
    idx_free = wi.get('free') if wi else None
    if idx_free:
        seq = np.zeros((1, MAX_SEQ_LEN), dtype=np.int32)
        seq[0, 0] = idx_free
        print("Predicción con token 'free' en la primera posición:", float(model.predict(seq)[0][0]))

def main():
    model, tokenizer = load_resources()
    print('\nModel summary:')
    model.summary()
    inspect_last_layer(model)
    try_examples(model, tokenizer)

if __name__ == '__main__':
    main()
