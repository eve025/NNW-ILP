"""
Script para entrenar un LSTM detector de spam.
Guarda el modelo en `spam_lstm_model.h5` y el tokenizer en `tokenizer.pkl`.
Uso: python train.py
"""
import os
import random
import re
import zipfile
import requests
import io
import pickle
import glob
import unicodedata

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils import class_weight, resample

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Input

# NLP utils
import nltk
from nltk.corpus import stopwords
try:
    from unidecode import unidecode
except Exception:
    unidecode = None
import matplotlib.pyplot as plt

# Parámetros
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
WORKDIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(WORKDIR, "spam_lstm_model.h5")
TOKENIZER_PATH = os.path.join(WORKDIR, "tokenizer.pkl")

MAX_VOCAB_SIZE = 10000
MAX_SEQ_LEN = 100
EMBED_DIM = 100

# Preparar stopwords en español (descarga si es necesario)
nltk.download('stopwords', quiet=True)
SPANISH_STOPWORDS = set(stopwords.words('spanish'))


def download_uci():
    print("Descargando dataset UCI SMS Spam Collection...")
    r = requests.get(DATA_URL)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    with z.open('SMSSpamCollection') as f:
        df = pd.read_csv(f, sep='\t', header=None, names=['label', 'message'])
    return df


def load_data():
    """Carga CSVs desde data/*.csv si existen; si no, descarga UCI como fallback.
    Espera columnas: 'label' y 'message' (si no, intenta usar las dos primeras columnas).
    """
    data_dir = os.path.join(WORKDIR, 'data')
    files = glob.glob(os.path.join(data_dir, '*.csv'))
    if files:
        print(f"Cargando {len(files)} archivos desde {data_dir}...")
        dfs = []
        for fpath in files:
            try:
                d = pd.read_csv(fpath)
            except Exception:
                print(f"Error leyendo {fpath}, saltando.")
                continue
            if 'label' in d.columns and 'message' in d.columns:
                dfs.append(d[['label', 'message']])
            else:
                # intentar usar las dos primeras columnas
                cols = list(d.columns)
                if len(cols) >= 2:
                    tmp = d.iloc[:, :2].copy()
                    # Detectar automáticamente cuál columna parece contener la etiqueta
                    # Heurística: la columna de label suele tener pocas categorías (spam/ham/0/1)
                    def looks_like_label(series):
                        vals = series.dropna().astype(str).str.lower().str.strip()
                        unique_vals = set(vals.unique())
                        # valores típicos de etiquetas
                        common_labels = {'spam', 'ham', 's', 'h', '0', '1', 'true', 'false', 'yes', 'no'}
                        if len(unique_vals) <= 10 and len(unique_vals & common_labels) > 0:
                            return True
                        # si muy pocas categorías y cadenas cortas, probablemente sea label
                        if len(unique_vals) <= 10 and vals.str.len().mean() < 6:
                            return True
                        return False

                    col0_is_label = looks_like_label(tmp.iloc[:, 0])
                    col1_is_label = looks_like_label(tmp.iloc[:, 1])
                    if col0_is_label and not col1_is_label:
                        tmp.columns = ['label', 'message']
                    elif col1_is_label and not col0_is_label:
                        tmp = tmp.iloc[:, [1, 0]].copy()
                        tmp.columns = ['label', 'message']
                    else:
                        # fallback: si no está claro, intentar detectar por longitud de texto
                        len0 = tmp.iloc[:, 0].astype(str).str.len().median()
                        len1 = tmp.iloc[:, 1].astype(str).str.len().median()
                        if len0 < len1:
                            # 0 es probablemente label
                            tmp.columns = ['label', 'message']
                        else:
                            tmp = tmp.iloc[:, [1, 0]].copy()
                            tmp.columns = ['label', 'message']
                    dfs.append(tmp)
        if dfs:
            df = pd.concat(dfs, ignore_index=True)
            print(f"Mensajes cargados desde CSV: {len(df)}")
            return df
        else:
            print("No se pudieron leer archivos CSV válidos en data/; se usará dataset UCI.")
    # fallback
    df = download_uci()
    print(f"Mensajes cargados: {len(df)}")
    return df


def download_spanish_datasets():
    """Descarga datasets conocidos en español al directorio data/ si no existen.
    URLs añadidas por el usuario:
      - HuggingFace: softecapps/spam_ham_spanish (train.csv / test.csv)
      - GitHub: starzomee/Email-Spam-Detection (mail_data.csv)
    """
    data_dir = os.path.join(WORKDIR, 'data')
    os.makedirs(data_dir, exist_ok=True)

    sources = [
        {
            'url': 'https://huggingface.co/datasets/softecapps/spam_ham_spanish/resolve/main/train.csv',
            'out': os.path.join(data_dir, 'hf_train.csv')
        },
        {
            'url': 'https://huggingface.co/datasets/softecapps/spam_ham_spanish/resolve/main/test.csv',
            'out': os.path.join(data_dir, 'hf_test.csv')
        },
        {
            'url': 'https://raw.githubusercontent.com/starzomee/Email-Spam-Detection/main/mail_data.csv',
            'out': os.path.join(data_dir, 'mail_data.csv')
        }
    ]

    for s in sources:
        out_path = s['out']
        if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            print(f"Ya existe {os.path.basename(out_path)}, se omite descarga.")
            continue
        try:
            print(f"Descargando {s['url']} -> {out_path} ...")
            r = requests.get(s['url'], timeout=30)
            r.raise_for_status()
            with open(out_path, 'wb') as f:
                f.write(r.content)
            print(f"Descargado {out_path}")
        except Exception as e:
            print(f"No se pudo descargar {s['url']}: {e}")


def remove_accents(text):
    if unidecode:
        return unidecode(text)
    # fallback
    text = unicodedata.normalize('NFKD', text)
    return ''.join([c for c in text if not unicodedata.combining(c)])


def clean_text(text):
    # lowercase + quitar acentos
    text = str(text).lower()
    text = remove_accents(text)
    text = re.sub(r'http\S+|www\S+', ' ', text)
    text = re.sub(r'\S+@\S+', ' ', text)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    # normalizar espacios
    text = re.sub(r'\s+', ' ', text).strip()
    # remover stopwords en español (si están)
    if SPANISH_STOPWORDS:
        tokens = [w for w in text.split() if w not in SPANISH_STOPWORDS]
        return ' '.join(tokens)
    return text


def prepare_tokenizer(texts, max_words=MAX_VOCAB_SIZE):
    tokenizer = Tokenizer(num_words=max_words, oov_token='<OOV>')
    tokenizer.fit_on_texts(texts)
    return tokenizer


def build_model(vocab_size, embed_dim=EMBED_DIM, input_length=MAX_SEQ_LEN):
    # Use input_shape instead of deprecated input_length; build model explicitly
    model = Sequential([
        Input(shape=(input_length,)),
        Embedding(vocab_size, embed_dim),
        LSTM(128, return_sequences=False),
        Dropout(0.4),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def map_label_to_num(x):
    # Normalizar etiquetas comunes a 0 (no spam) / 1 (spam)
    s = str(x).strip().lower()
    if s in ['1', 'true', 't', 'yes', 'y', 'spam', 's']:
        return 1
    if s in ['0', 'false', 'f', 'no', 'n', 'ham', 'not spam', 'legit', 'legitimate']:
        return 0
    # heurística: si contiene 'spam'
    if 'spam' in s:
        return 1
    return 0


def main():
    df = load_data()
    # normalizar columnas si vienen con otros nombres
    if 'message' not in df.columns and 'text' in df.columns:
        df = df.rename(columns={'text':'message'})
    if 'label' not in df.columns and 'target' in df.columns:
        df = df.rename(columns={'target':'label'})

    df['clean'] = df['message'].astype(str).apply(clean_text)
    df['label_num'] = df['label'].apply(map_label_to_num)

    # Simple oversampling (upsample minoritaria) para mitigar desbalance extremo
    counts = df['label_num'].value_counts()
    if len(counts) > 1 and counts.min() < counts.max():
        print("Desbalance detectado en labels:")
        print(counts.to_dict())
        # separar mayoritaria y minoritaria
        maj = df[df['label_num'] == counts.idxmax()]
        mino = df[df['label_num'] == counts.idxmin()]
        # upsample minoritaria hasta igualar la mayoritaria
        mino_upsampled = resample(mino, replace=True, n_samples=len(maj), random_state=RANDOM_SEED)
        df = pd.concat([maj, mino_upsampled]).sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
        print(f"Después de upsampling: distribución -> {df['label_num'].value_counts().to_dict()}")

    X = df['clean'].values
    y = df['label_num'].values

    tokenizer = prepare_tokenizer(X)
    sequences = tokenizer.texts_to_sequences(X)
    X_pad = pad_sequences(sequences, maxlen=MAX_SEQ_LEN, padding='post')

    X_train, X_test, y_train, y_test = train_test_split(X_pad, y, test_size=0.2, random_state=RANDOM_SEED)
    # separar un conjunto de validación desde el training set (10%) para control y selección de umbral
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=RANDOM_SEED)

    vocab_size = min(MAX_VOCAB_SIZE, len(tokenizer.word_index) + 1)
    # ensure vocab_size at least 2 to avoid zero-parameter Embedding
    if vocab_size < 2:
        vocab_size = 2
    # imprimir información útil para depuración
    print(f"vocab_size = {vocab_size}")
    unique, counts = np.unique(y, return_counts=True)
    print("Distribución de clases en el dataset:")
    for u, c in zip(unique, counts):
        print(f"  clase {u}: {c} ejemplos")
    model = build_model(vocab_size)
    # build the model to populate shapes and parameter counts (helps summary)
    try:
        model.build((None, MAX_SEQ_LEN))
    except Exception:
        pass
    model.summary()

    print("Entrenando el modelo...")
    # calcular class weights para balancear la pérdida frente al desequilibrio de clases
    try:
        cw = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
        class_weights = {i: float(w) for i, w in enumerate(cw)}
        print(f"Usando class_weights: {class_weights}")
    except Exception as e:
        print(f"No se pudo calcular class_weight: {e}")
        class_weights = None

    # Aumentamos epochs para dar más oportunidad al modelo de aprender
    # callbacks: guardar el mejor modelo según val_loss
    from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
    ckpt = ModelCheckpoint(MODEL_PATH, monitor='val_loss', save_best_only=True, verbose=1)
    early = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=False, verbose=1)

    history = model.fit(
        X_train, y_train,
        epochs=10,
        batch_size=64,
        validation_data=(X_val, y_val),
        class_weight=class_weights,
        callbacks=[ckpt, early]
    )

    print("Evaluando en test set...")
    # cargar el mejor modelo guardado por el callback (si existe)
    try:
        best_model = load_model(MODEL_PATH)
        print('Cargado mejor modelo desde', MODEL_PATH)
    except Exception:
        best_model = model

    # evaluar en test set usando umbral por defecto 0.5 (se puede actualizar)
    preds = (best_model.predict(X_test) > 0.5).astype(int).flatten()
    print(classification_report(y_test, preds, target_names=['no spam','spam'], zero_division=0))
    print("Accuracy:", accuracy_score(y_test, preds))

    # calcular mejor umbral en la validación (maximizar F1)
    try:
        y_val_probs = best_model.predict(X_val).ravel()
        precision, recall, thresholds = precision_recall_curve(y_val, y_val_probs)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        best_idx = np.nanargmax(f1_scores[:-1]) if len(f1_scores) > 1 else 0
        best_thresh = thresholds[best_idx] if len(thresholds) > 0 else 0.5
        print(f"Mejor umbral en validación (F1): {best_thresh:.4f}  F1: {f1_scores[best_idx]:.4f}")
        # guardar umbral
        thresh_path = os.path.join(WORKDIR, 'best_threshold.txt')
        with open(thresh_path, 'w', encoding='utf-8') as tfp:
            tfp.write(str(best_thresh))
        print('Umbral guardado en', thresh_path)
    except Exception as e:
        print('No se pudo calcular umbral:', e)

    # Guardar reporte de métricas
    report_path = os.path.join(WORKDIR, 'training_report.txt')
    with open(report_path, 'w', encoding='utf-8') as rf:
        rf.write('Classification report:\n')
        rf.write(classification_report(y_test, preds, target_names=['no spam','spam'], zero_division=0))
        rf.write('\n\n')
        rf.write('Confusion matrix:\n')
        rf.write(str(confusion_matrix(y_test, preds)))
        rf.write('\n\n')
        rf.write(f'Accuracy: {accuracy_score(y_test, preds)}\n')
        # distribución de clases
        unique, counts = np.unique(y, return_counts=True)
        rf.write('\nDataset class distribution:\n')
        for u, c in zip(unique, counts):
            rf.write(f'{u}: {c}\n')
    print(f"Reporte guardado en {report_path}")

    # Guardar curva de entrenamiento
    try:
        plt.figure(figsize=(8,4))
        plt.subplot(1,2,1)
        plt.plot(history.history['loss'], label='train_loss')
        plt.plot(history.history['val_loss'], label='val_loss')
        plt.legend(); plt.title('Loss')
        plt.subplot(1,2,2)
        plt.plot(history.history['accuracy'], label='train_acc')
        plt.plot(history.history['val_accuracy'], label='val_acc')
        plt.legend(); plt.title('Accuracy')
        plot_path = os.path.join(WORKDIR, 'training_plot.png')
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()
        print(f"Gráfica de entrenamiento guardada en {plot_path}")
    except Exception as e:
        print(f"No se pudo guardar la gráfica: {e}")

    print(f"Guardando modelo en {MODEL_PATH}")
    model.save(MODEL_PATH)
    print(f"Guardando tokenizer en {TOKENIZER_PATH}")
    with open(TOKENIZER_PATH, 'wb') as f:
        pickle.dump(tokenizer, f)

    print("Listo. Usa el notebook para más análisis y visualizaciones.")


if __name__ == '__main__':
    main()
