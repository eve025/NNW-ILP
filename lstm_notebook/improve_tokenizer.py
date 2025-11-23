#!/usr/bin/env python3
"""Build improved tokenizer(s) from available normalized CSVs.

This script reads `lstm_notebook/data/normalized_*.csv` and trains a
Keras Tokenizer with `oov_token` and lowercasing. It saves tokenizer files
as `tokenizer_<lang>.pkl` and `tokenizer.pkl` (fallback).

Usage:
  py -3 improve_tokenizer.py --max_words 20000 --lang es
"""
import argparse
from pathlib import Path
import pickle
import pandas as pd

BASE = Path(__file__).resolve().parent
DATA = BASE / 'data'


def read_texts(lang=None):
    patterns = list(DATA.glob('normalized_*.csv'))
    texts = []
    for p in patterns:
        try:
            df = pd.read_csv(p, encoding='utf8', usecols=lambda c: c.lower() in ['label','message','text'])
            if 'message' in df.columns:
                texts += df['message'].astype(str).tolist()
            elif 'text' in df.columns:
                texts += df['text'].astype(str).tolist()
            else:
                texts += df.iloc[:, -1].astype(str).tolist()
        except Exception:
            continue
    return texts


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--max_words', type=int, default=10000)
    p.add_argument('--oov_token', default='<OOV>')
    args = p.parse_args()

    texts = read_texts()
    if not texts:
        raise SystemExit('No texts found in data folder')

    # train tokenizer
    from tensorflow.keras.preprocessing.text import Tokenizer
    tok = Tokenizer(num_words=args.max_words, oov_token=args.oov_token, lower=True)
    tok.fit_on_texts(texts)

    # save
    with open(BASE / 'tokenizer.pkl', 'wb') as fh:
        pickle.dump(tok, fh)
    print('Saved tokenizer to', BASE / 'tokenizer.pkl')


if __name__ == '__main__':
    main()
