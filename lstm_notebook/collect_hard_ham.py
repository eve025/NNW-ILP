#!/usr/bin/env python3
"""Collect ham examples that the model currently misclassifies as spam.

This script reads one or more input files (plain text with one message per line
or CSV with 'label,message'), filters ham rows where the current model predicts
spam (prob >= threshold) and writes them to `lstm_notebook/data/hard_ham.csv`.

Usage:
  py -3 collect_hard_ham.py --inputs data/extra_ham.txt data/more.csv --threshold 0.42
"""
import argparse
import os
import csv
import json
import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parent
OUT = BASE / 'data' / 'hard_ham.csv'


def read_messages(paths):
    rows = []
    for p in paths:
        p = Path(p)
        if not p.exists():
            continue
        if p.suffix.lower() in ['.csv']:
            try:
                df = pd.read_csv(p, encoding='utf8', usecols=lambda c: c.lower() in ['label', 'message', 'text'], on_bad_lines='skip')
                # try common columns
                if 'message' in df.columns:
                    texts = df['message'].astype(str).tolist()
                elif 'text' in df.columns:
                    texts = df['text'].astype(str).tolist()
                else:
                    # fallback: take first column
                    texts = df.iloc[:, -1].astype(str).tolist()
                # determine label if present
                if 'label' in df.columns:
                    labels = df['label'].astype(str).tolist()
                    rows += list(zip(labels, texts))
                else:
                    rows += [('ham', t) for t in texts]
            except Exception:
                continue
        else:
            # plain text file
            with p.open('r', encoding='utf8', errors='ignore') as fh:
                for l in fh:
                    t = l.strip()
                    if t:
                        rows.append(('ham', t))
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--inputs', nargs='+', required=True)
    p.add_argument('--model', default=str(BASE / 'spam_lstm_model.h5'))
    p.add_argument('--tokenizer', default=str(BASE / 'tokenizer.pkl'))
    p.add_argument('--threshold', type=float, default=0.42)
    args = p.parse_args()

    # lazy imports to fail early only when needed
    import pickle
    import tensorflow as tf
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    # load tokenizer and model
    if not Path(args.tokenizer).exists():
        raise SystemExit('Tokenizer not found: ' + args.tokenizer)
    with open(args.tokenizer, 'rb') as fh:
        tokenizer = pickle.load(fh)
    if not Path(args.model).exists():
        raise SystemExit('Model not found: ' + args.model)
    # Try to load full model; if it fails due to a custom layer (e.g. NotEqual),
    # try to rebuild architecture via retrain_with_hard_negatives.build_model()
    try:
        model = tf.keras.models.load_model(args.model)
    except Exception as e:
        # fallback: try to load weights into known architecture
        print('Warning: load_model failed, attempting to rebuild architecture and load weights:', e)
        try:
            from retrain_with_hard_negatives import build_model
            weights_path = Path(args.model).with_suffix('.weights.h5')
            if not weights_path.exists():
                # Accept also filenames like spam_lstm_model_refit.weights.h5
                alt = Path(args.model).parent / (Path(args.model).stem + '.weights.h5')
                if alt.exists():
                    weights_path = alt
            if not weights_path.exists():
                raise SystemExit('Weights snapshot not found: ' + str(weights_path))
            model = build_model()
            model.load_weights(str(weights_path))
            print('Loaded model architecture and weights from', weights_path)
        except Exception as e2:
            raise

    rows = read_messages(args.inputs)
    if not rows:
        print('No inputs read')
        return

    texts = [r[1] for r in rows]
    seqs = tokenizer.texts_to_sequences(texts)
    X = pad_sequences(seqs, maxlen=100, padding='post')
    probs = model.predict(X, batch_size=64).ravel()

    hard = []
    for (label, text), p in zip(rows, probs):
        if label.lower().startswith('h') and p >= args.threshold:
            hard.append({'message': text, 'prob': float(p)})

    OUT.parent.mkdir(parents=True, exist_ok=True)
    # save CSV
    with OUT.open('w', encoding='utf8', newline='') as fh:
        writer = csv.writer(fh)
        writer.writerow(['message', 'prob'])
        for h in hard:
            writer.writerow([h['message'], h['prob']])

    print(f'Wrote {len(hard)} hard ham examples to {OUT}')


if __name__ == '__main__':
    main()
