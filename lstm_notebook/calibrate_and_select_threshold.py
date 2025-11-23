#!/usr/bin/env python3
"""Calibrate model probabilities (isotonic) and select threshold per language.

Reads `lstm_notebook/data/normalized_<lang>.csv` for each language found,
computes model probabilities, fits an isotonic calibrator using validation
split, writes calibrated CSV and recommended threshold to disk.

Usage:
  py -3 calibrate_and_select_threshold.py --model spam_lstm_model.h5 --tokenizer tokenizer.pkl
"""
import argparse
from pathlib import Path
import pandas as pd
import pickle
import numpy as np

BASE = Path(__file__).resolve().parent
DATA = BASE / 'data'


def load_data(lang):
    p = DATA / f'normalized_{lang}.csv'
    if not p.exists():
        return None
    df = pd.read_csv(p, encoding='utf8', on_bad_lines='skip')
    # try to locate message and label
    col_text = None
    for c in df.columns:
        if c.lower() in ('message','text','body'):
            col_text = c
            break
    if col_text is None:
        col_text = df.columns[-1]
    # label column
    col_label = None
    for c in df.columns:
        if c.lower() in ('label','y'):
            col_label = c
            break
    if col_label is None:
        # assume all are ham
        df['label'] = 'ham'
        col_label = 'label'
    df = df[[col_label, col_text]].dropna()
    df.columns = ['label','message']
    return df


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default=str(BASE / 'spam_lstm_model.h5'))
    p.add_argument('--tokenizer', default=str(BASE / 'tokenizer.pkl'))
    p.add_argument('--lang', choices=['es','en','all'], default='all')
    args = p.parse_args()

    import tensorflow as tf
    from sklearn.isotonic import IsotonicRegression
    from sklearn.model_selection import train_test_split
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    with open(args.tokenizer, 'rb') as fh:
        tok = pickle.load(fh)

    # Try loading the full model; if it fails due to a custom 'NotEqual' layer,
    # register a minimal fallback implementation to allow inference for calibration.
    # If a weights-only snapshot exists, rebuild the architecture and load weights.
    weights_path = BASE / 'spam_lstm_model_refit.weights.h5'
    if weights_path.exists():
        try:
            from retrain_with_hard_negatives import build_model as _build_model
            model = _build_model()
            model.load_weights(str(weights_path))
            print('Loaded model architecture and weights from', weights_path)
        except Exception as e:
            print('Failed to load weights snapshot, falling back to load_model():', e)
            try:
                model = tf.keras.models.load_model(args.model, compile=False)
            except ValueError as e:
                msg = str(e)
                if 'NotEqual' in msg:
                    class NotEqual(tf.keras.layers.Layer):
                        def __init__(self, *args, **kwargs):
                            super().__init__()

                        @classmethod
                        def from_config(cls, config):
                            return cls()

                        def call(self, inputs, **kwargs):
                            import tensorflow as _tf
                            if isinstance(inputs, (list, tuple)) and len(inputs) >= 2:
                                return _tf.not_equal(inputs[0], inputs[1])
                            if isinstance(inputs, (list, tuple)) and len(inputs) == 1:
                                return _tf.not_equal(inputs[0], 0)
                            return _tf.not_equal(inputs, 0)

                        def get_config(self):
                            return {}

                    model = tf.keras.models.load_model(args.model, custom_objects={'NotEqual': NotEqual}, compile=False)
                else:
                    raise
    else:
        try:
            model = tf.keras.models.load_model(args.model, compile=False)
        except ValueError as e:
            msg = str(e)
            if 'NotEqual' in msg:
                class NotEqual(tf.keras.layers.Layer):
                    def __init__(self, *args, **kwargs):
                        super().__init__()

                    @classmethod
                    def from_config(cls, config):
                        return cls()

                    def call(self, inputs, **kwargs):
                        import tensorflow as _tf
                        if isinstance(inputs, (list, tuple)) and len(inputs) >= 2:
                            return _tf.not_equal(inputs[0], inputs[1])
                        if isinstance(inputs, (list, tuple)) and len(inputs) == 1:
                            return _tf.not_equal(inputs[0], 0)
                        return _tf.not_equal(inputs, 0)

                    def get_config(self):
                        return {}

                model = tf.keras.models.load_model(args.model, custom_objects={'NotEqual': NotEqual}, compile=False)
            else:
                raise

    if args.lang == 'all':
        # discover all files named normalized_*.csv in the data folder
        langs = []
        for p in sorted(DATA.glob('normalized_*.csv')):
            name = p.name
            key = name[len('normalized_'):-4]
            if key and key not in langs:
                langs.append(key)
    else:
        langs = [args.lang]
    for lang in langs:
        df = load_data(lang)
        if df is None or df.empty:
            print('No data for', lang)
            continue
        # create binary label
        y = (df['label'].str.lower().str.startswith('s')).astype(int)
        Xseq = tok.texts_to_sequences(df['message'].astype(str).tolist())
        X = pad_sequences(Xseq, maxlen=100, padding='post')

        # train/val split for calibration
        Xtr, Xval, ytr, yval = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)

        probs_val = model.predict(Xval, batch_size=128).ravel()

        # fit isotonic
        iso = IsotonicRegression(out_of_bounds='clip')
        try:
            iso.fit(probs_val, yval)
        except Exception as e:
            print('Isotonic fit failed for', lang, e)
            continue

        calibrated = iso.transform(probs_val)

        # evaluate thresholds and pick one with precision >=0.9 maximizing recall
        thresholds = np.linspace(0.01, 0.99, 99)
        rows = []
        for t in thresholds:
            preds = (calibrated >= t).astype(int)
            tp = int((preds & yval).sum())
            fp = int((preds & (1 - yval)).sum())
            fn = int(((1 - preds) & yval).sum())
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            rows.append((t, prec, rec))

        df_thresh = pd.DataFrame(rows, columns=['threshold','precision','recall'])
        out_csv = BASE / f'retrain_by_lang_thresholds_{lang}_calibrated.csv'
        df_thresh.to_csv(out_csv, index=False)

        # pick recommended threshold
        cand = df_thresh[df_thresh['precision'] >= 0.9]
        if not cand.empty:
            # choose one with max recall
            best = cand.loc[cand['recall'].idxmax()]
            recommended = float(best['threshold'])
        else:
            # fallback: threshold with best f1
            df_thresh['f1'] = 2 * (df_thresh['precision'] * df_thresh['recall']) / (df_thresh['precision'] + df_thresh['recall'] + 1e-9)
            best = df_thresh.loc[df_thresh['f1'].idxmax()]
            recommended = float(best['threshold'])

        # save calibrator and recommendation
        with open(BASE / f'calibrator_{lang}.pkl', 'wb') as fh:
            pickle.dump(iso, fh)
        with open(BASE / f'recommended_threshold_{lang}.txt', 'w', encoding='utf8') as fh:
            fh.write(str(recommended))

        print('Lang', lang, 'recommended threshold', recommended, 'wrote', out_csv)


if __name__ == '__main__':
    main()
