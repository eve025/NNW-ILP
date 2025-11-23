#!/usr/bin/env python3
r"""Hybrid inference: model + heuristic rules + whitelist.

Usage examples:
    # single message
    py -3 ./lstm_notebook/hybrid_policy.py --message "Revisa tu factura" --lang es

    # file with one message per line
    py -3 ./lstm_notebook/hybrid_policy.py --file messages.txt --lang en

The classifier returns: probability, model_decision (prob>=threshold), final_decision (after rules/whitelist)
"""
import os
import re
import argparse
import json
import pickle
from urllib.parse import urlparse
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL = os.path.join(BASE, 'spam_lstm_model.h5')
DEFAULT_TOKENIZER = os.path.join(BASE, 'tokenizer.pkl')
WHITELIST_FILE = os.path.join(BASE, 'whitelist_phrases.txt')
DEFAULT_THRESHOLD = 0.42


def load_whitelist(path=WHITELIST_FILE):
    if not os.path.exists(path):
        return []
    with open(path, 'r', encoding='utf8') as fh:
        phrases = [l.strip().lower() for l in fh if l.strip() and not l.strip().startswith('#')]
    return phrases


def has_url(text):
    return bool(re.search(r'https?://|www\.|\.[a-z]{2,3}/', text))


def has_money_keyword(text):
    kws = ['pago', 'factura', 'comprobante', 'transferencia', '$', 'mxn', 'usd', 'precio']
    t = text.lower()
    return any(k in t for k in kws)


def matches_whitelist(text, whitelist):
    t = text.lower()
    return any(p in t for p in whitelist)


def load_tokenizer(tok_path=DEFAULT_TOKENIZER):
    if os.path.exists(tok_path):
        with open(tok_path, 'rb') as fh:
            return pickle.load(fh)
    return None


def tokenize_text(tokenizer, texts, maxlen=100):
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    seqs = tokenizer.texts_to_sequences(texts)
    return pad_sequences(seqs, maxlen=maxlen, padding='post')


def load_model(path=DEFAULT_MODEL):
    import tensorflow as tf
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return tf.keras.models.load_model(path)


def hybrid_decision(text, model, tokenizer, whitelist, threshold=DEFAULT_THRESHOLD):
    clean = str(text).strip()
    probs = model.predict(tokenize_text(tokenizer, [clean]))[0][0]
    model_decision = bool(probs >= threshold)

    # Whitelist overrides (if message matches whitelist, force ham)
    if matches_whitelist(clean, whitelist):
        return float(probs), model_decision, 'ham (whitelist)'

    # Heuristic rules to reduce false positives: require both model and heuristics for spam
    # If model strongly predicts spam (prob >> threshold), accept spam
    if probs >= max(0.8, threshold + 0.3):
        return float(probs), model_decision, 'spam (high-prob)'

    # Otherwise require model_decision and presence of spam indicators
    indicators = has_url(clean) or has_money_keyword(clean)
    if model_decision and indicators:
        return float(probs), model_decision, 'spam (model+indicators)'

    # Default: ham
    return float(probs), model_decision, 'ham (default)'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--message', help='Single message to classify')
    p.add_argument('--file', help='File with one message per line')
    p.add_argument('--model', help='Path to model', default=DEFAULT_MODEL)
    p.add_argument('--tokenizer', help='Path to tokenizer', default=DEFAULT_TOKENIZER)
    p.add_argument('--threshold', type=float, default=None)
    p.add_argument('--lang', choices=['es','en','other'], default='es')
    args = p.parse_args()

    tokenizer = load_tokenizer(args.tokenizer)
    if tokenizer is None:
        raise SystemExit('Tokenizer not found at ' + args.tokenizer)
    model = load_model(args.model)
    whitelist = load_whitelist()
    threshold = args.threshold if args.threshold is not None else DEFAULT_THRESHOLD

    messages = []
    if args.message:
        messages = [args.message]
    elif args.file:
        with open(args.file, 'r', encoding='utf8') as fh:
            messages = [l.strip() for l in fh if l.strip()]
    else:
        raise SystemExit('Provide --message or --file')

    for m in messages:
        prob, mdl, final = hybrid_decision(m, model, tokenizer, whitelist, threshold=threshold)
        print(json.dumps({'message': m, 'prob': prob, 'model_decision': bool(mdl), 'final_decision': final}, ensure_ascii=False))


if __name__ == '__main__':
    main()
