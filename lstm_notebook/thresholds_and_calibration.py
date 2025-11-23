import os
import glob
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.isotonic import IsotonicRegression
import tensorflow as tf

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_GLOB = os.path.join(BASE, 'data', 'normalized_*.csv')
MAX_SEQ_LEN = 100
MAX_VOCAB_SIZE = 10000

def detect_lang(text):
    t = str(text).lower()
    SPANISH_KEYWORDS = set([' que ',' de ',' la ',' el ',' los ',' las ',' para ',' por ','gracias','hola','mañana','cita','usted','tu','favor'])
    if any(ch in t for ch in 'áéíóúñ') or any(k in t for k in SPANISH_KEYWORDS):
        return 'es'
    if any(k in t for k in [' the ',' you ',' thanks','meeting','password','please','tomorrow']):
        return 'en'
    return 'other'

def clean_text(t):
    import unicodedata, re
    s = str(t).lower()
    s = unicodedata.normalize('NFKD', s).encode('ascii','ignore').decode('ascii')
    s = re.sub(r'http\S+|www\S+',' ', s)
    s = re.sub(r'\S+@\S+',' ', s)
    s = re.sub(r'[^a-z0-9\s]',' ', s)
    s = re.sub(r'\s+',' ', s).strip()
    return s

def load_data():
    files = sorted(glob.glob(DATA_GLOB))
    if not files:
        raise SystemExit('No normalized CSVs found in data/')
    frames = []
    for p in files:
        try:
            df = pd.read_csv(p)
        except Exception:
            rows = []
            with open(p, 'r', encoding='utf8', errors='ignore') as fh:
                for ln in fh:
                    ln = ln.strip('\n')
                    if not ln: continue
                    if ',' in ln:
                        a,b = ln.split(',',1)
                    else:
                        parts = ln.split('\t')
                        if len(parts) >=2:
                            a,b = parts[0], parts[1]
                        else:
                            continue
                    rows.append((a.strip(), b.strip()))
            df = pd.DataFrame(rows, columns=['label','message'])
        if 'label' not in df.columns or 'message' not in df.columns:
            df = df.iloc[:, :2]
            df.columns = ['label','message']
        frames.append(df)
    all_df = pd.concat(frames, ignore_index=True)
    all_df['label_num'] = all_df['label'].apply(lambda v: 1 if str(v).strip().lower() in ('spam','s','1','true','yes') else 0)
    all_df['message'] = all_df['message'].astype(str)
    all_df['lang'] = all_df['message'].apply(detect_lang)
    return all_df

def prepare_lang_df(all_df, lang):
    df = all_df[all_df['lang']==lang].copy()
    df['clean'] = df['message'].apply(clean_text)
    return df

def tokenize_with_saved(tokenizer_path, texts):
    if os.path.exists(tokenizer_path):
        with open(tokenizer_path, 'rb') as fh:
            tokenizer = pickle.load(fh)
    else:
        from tensorflow.keras.preprocessing.text import Tokenizer
        tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE, oov_token='<OOV>')
        tokenizer.fit_on_texts(texts)
    seqs = tokenizer.texts_to_sequences(texts)
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    X = pad_sequences(seqs, maxlen=MAX_SEQ_LEN, padding='post')
    return tokenizer, X

def sweep_thresholds(probs, y_true, thresholds):
    rows = []
    for thr in thresholds:
        preds = (probs >= thr).astype(int)
        p = precision_score(y_true, preds, zero_division=0)
        r = recall_score(y_true, preds, zero_division=0)
        f = f1_score(y_true, preds, zero_division=0)
        a = accuracy_score(y_true, preds)
        rows.append({'threshold': float(thr), 'precision': float(p), 'recall': float(r), 'f1': float(f), 'accuracy': float(a)})
    return pd.DataFrame(rows)

def run():
    all_df = load_data()
    counts = all_df['lang'].value_counts().to_dict()
    print('Language counts:', counts)
    results = {}
    thresholds = np.linspace(0.01,0.99,99)
    for lang in ['es','en']:
        df_lang = prepare_lang_df(all_df, lang)
        if len(df_lang) < 50:
            print(f'skipping {lang}, too few samples')
            continue
        X = df_lang['clean'].values
        y = df_lang['label_num'].values
        # stratified split: test 15%, then val 10% of train
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42, stratify=y_train)

        tok_path = os.path.join(BASE, f'tokenizer_{lang}.pkl')
        tokenizer, X_tr = tokenize_with_saved(tok_path, X_train)
        _, X_v = tokenize_with_saved(tok_path, X_val)
        _, X_te = tokenize_with_saved(tok_path, X_test)

        # load model
        model_path = os.path.join(BASE, f'spam_lstm_model_{lang}.h5')
        if not os.path.exists(model_path):
            model_path = os.path.join(BASE, 'spam_lstm_model.h5')
        model = tf.keras.models.load_model(model_path)

        probs_val = model.predict(X_v).flatten()
        probs_test = model.predict(X_te).flatten()

        df_thresh = sweep_thresholds(probs_test, y_test, thresholds)
        out_csv = os.path.join(BASE, f'retrain_by_lang_thresholds_{lang}.csv')
        df_thresh.to_csv(out_csv, index=False)

        # plot
        plt.figure(figsize=(8,5))
        plt.plot(df_thresh['threshold'], df_thresh['precision'], label='precision')
        plt.plot(df_thresh['threshold'], df_thresh['recall'], label='recall')
        plt.plot(df_thresh['threshold'], df_thresh['f1'], label='f1')
        plt.plot(df_thresh['threshold'], df_thresh['accuracy'], label='accuracy')
        plt.xlabel('threshold')
        plt.legend()
        plt.title(f'Metrics vs threshold ({lang})')
        png_path = os.path.join(BASE, f'thresholds_{lang}.png')
        plt.tight_layout()
        plt.savefig(png_path)
        plt.close()

        # find suggested threshold achieving precision >= target_precision if possible
        try:
            target_precision = float(os.environ.get('TARGET_PRECISION', '0.90'))
        except Exception:
            target_precision = 0.90
        candidates = df_thresh[df_thresh['precision'] >= target_precision]
        if len(candidates):
            best = candidates.loc[candidates['recall'].idxmax()]
            suggested = float(best['threshold'])
        else:
            best = df_thresh.loc[df_thresh['f1'].idxmax()]
            suggested = float(best['threshold'])

        # Calibration isotonic on val
        try:
            iso = IsotonicRegression(out_of_bounds='clip')
            iso.fit(probs_val, y_val)
            probs_test_cal = iso.transform(probs_test)
            df_thresh_cal = sweep_thresholds(probs_test_cal, y_test, thresholds)
            out_csv_cal = os.path.join(BASE, f'retrain_by_lang_thresholds_{lang}_calibrated.csv')
            df_thresh_cal.to_csv(out_csv_cal, index=False)
            # plot calibrated
            plt.figure(figsize=(8,5))
            plt.plot(df_thresh_cal['threshold'], df_thresh_cal['precision'], label='precision')
            plt.plot(df_thresh_cal['threshold'], df_thresh_cal['recall'], label='recall')
            plt.plot(df_thresh_cal['threshold'], df_thresh_cal['f1'], label='f1')
            plt.plot(df_thresh_cal['threshold'], df_thresh_cal['accuracy'], label='accuracy')
            plt.xlabel('threshold')
            plt.legend()
            plt.title(f'Metrics vs threshold ({lang}) - calibrated')
            png_cal = os.path.join(BASE, f'thresholds_{lang}_calibrated.png')
            plt.tight_layout()
            plt.savefig(png_cal)
            plt.close()
        except Exception as e:
            print('Calibration failed for', lang, e)
            df_thresh_cal = None

        results[lang] = {
            'samples': int(len(df_lang)),
            'train': int(len(X_tr)),
            'val': int(len(X_v)),
            'test': int(len(X_te)),
            'suggested_threshold': suggested,
            'thresholds_csv': out_csv,
            'thresholds_csv_calibrated': out_csv_cal if df_thresh_cal is not None else None,
            'plot': png_path,
            'plot_calibrated': png_cal if df_thresh_cal is not None else None
        }

    out = os.path.join(BASE, 'thresholds_and_calibration_summary.json')
    with open(out, 'w', encoding='utf8') as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)
    print('Saved summary to', out)

if __name__ == '__main__':
    run()
