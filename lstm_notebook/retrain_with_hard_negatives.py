#!/usr/bin/env python3
"""Reentrena el LSTM incorporando `normalized_extra.csv` y usando class_weight.

Genera: spam_lstm_model_refit.h5, tokenizer_refit.pkl, best_threshold_refit.txt, training_report_refit.txt

Este script es una versión controlada del reentreno pensada para pruebas rápidas (EPOCHS via env var).
"""
import os
import glob
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_GLOB = os.path.join(BASE, 'data', 'normalized_*.csv')
MAX_SEQ_LEN = 100
MAX_VOCAB = 10000
EMBED_DIM = 100


def load_all():
    files = sorted(glob.glob(DATA_GLOB))
    if not files:
        raise SystemExit('No data files found under data/*.csv')
    frames = [pd.read_csv(p) for p in files]
    df = pd.concat(frames, ignore_index=True)
    if 'label' not in df.columns or 'message' not in df.columns:
        df = df.iloc[:, :2]
        df.columns = ['label','message']
    df['label_num'] = df['label'].apply(lambda v: 1 if str(v).strip().lower() in ('spam','s','1','true','yes') else 0)
    df['message'] = df['message'].astype(str)
    return df


def clean_texts(texts):
    import unicodedata, re
    out = []
    for s in texts:
        t = str(s).lower()
        t = unicodedata.normalize('NFKD', t).encode('ascii','ignore').decode('ascii')
        t = re.sub(r'http\S+|www\S+',' ', t)
        t = re.sub(r'[^a-z0-9\s]',' ', t)
        t = re.sub(r'\s+',' ', t).strip()
        out.append(t)
    return out


def build_tokenizer(texts):
    from tensorflow.keras.preprocessing.text import Tokenizer
    tok = Tokenizer(num_words=MAX_VOCAB, oov_token='<OOV>')
    tok.fit_on_texts(texts)
    return tok


def texts_to_padded(tok, texts):
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    seqs = tok.texts_to_sequences(texts)
    return pad_sequences(seqs, maxlen=MAX_SEQ_LEN, padding='post')


def build_model():
    import tensorflow as tf
    from tensorflow.keras import layers, models
    inp = layers.Input(shape=(MAX_SEQ_LEN,))
    x = layers.Embedding(MAX_VOCAB, EMBED_DIM, mask_zero=True)(inp)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=False))(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    out = layers.Dense(1, activation='sigmoid')(x)
    model = models.Model(inp, out)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def main():
    import tensorflow as tf
    epochs = int(os.environ.get('EPOCHS', '3'))
    batch = int(os.environ.get('BATCH_SIZE', '64'))

    df = load_all()
    df['clean'] = clean_texts(df['message'].values)
    X = df['clean'].values
    y = df['label_num'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42, stratify=y_train)

    tokenizer = build_tokenizer(X_train)
    Xtr = texts_to_padded(tokenizer, X_train)
    Xv = texts_to_padded(tokenizer, X_val)
    Xte = texts_to_padded(tokenizer, X_test)

    classes = np.unique(y_train)
    cw = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weight = {int(c): float(w) for c, w in zip(classes, cw)}
    print('Class weight:', class_weight)

    model = build_model()
    model.fit(Xtr, y_train, epochs=epochs, batch_size=batch, validation_data=(Xv, y_val), class_weight=class_weight)

    # evaluate
    preds = (model.predict(Xte).flatten() >= 0.5).astype(int)
    report = classification_report(y_test, preds, output_dict=True)
    cm = confusion_matrix(y_test, preds)

    # save artifacts
    model_path = os.path.join(BASE, 'spam_lstm_model_refit.h5')
    tok_path = os.path.join(BASE, 'tokenizer_refit.pkl')
    model.save(model_path)
    # Also save weights-only snapshot so other scripts can rebuild the architecture
    weights_path = os.path.join(BASE, 'spam_lstm_model_refit.weights.h5')
    try:
        model.save_weights(weights_path)
    except Exception as e:
        print('Warning: failed to save weights-only snapshot:', e)
    with open(tok_path, 'wb') as fh:
        pickle.dump(tokenizer, fh)

    with open(os.path.join(BASE, 'training_report_refit.txt'), 'w', encoding='utf8') as fh:
        fh.write('Classification report:\n')
        fh.write(json.dumps(report, indent=2))
        fh.write('\nConfusion matrix:\n')
        fh.write(str(cm))

    # quick threshold sweep on val to suggest threshold
    from sklearn.metrics import precision_score, recall_score, f1_score
    probs_val = model.predict(Xv).flatten()
    best_thr = 0.5
    best_f1 = -1
    for thr in np.linspace(0.01, 0.99, 99):
        preds = (probs_val >= thr).astype(int)
        f = f1_score(y_val, preds, zero_division=0)
        if f > best_f1:
            best_f1 = f
            best_thr = float(thr)

    with open(os.path.join(BASE, 'best_threshold_refit.txt'), 'w', encoding='utf8') as fh:
        fh.write(str(best_thr))

    print('Saved model, tokenizer and best threshold:', best_thr)


if __name__ == '__main__':
    main()
