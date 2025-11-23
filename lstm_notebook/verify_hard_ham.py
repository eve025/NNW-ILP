#!/usr/bin/env python3
"""Filtra `data/hard_ham.csv` usando `whitelist_phrases.txt`.

Genera:
 - data/hard_ham_verified.csv (solo ejemplos no-whitelist)
 - data/normalized_hard_ham_verified.csv (misma data con nombre que recoge el reentreno)

Uso:
  py -3 verify_hard_ham.py --input data/hard_ham.csv --whitelist whitelist_phrases.txt --out data/hard_ham_verified.csv
"""
import argparse
from pathlib import Path
import csv


def load_whitelist(path):
    phrases = []
    p = Path(path)
    if not p.exists():
        return phrases
    with p.open('r', encoding='utf8', errors='ignore') as fh:
        for l in fh:
            t = l.strip()
            if not t or t.startswith('#'):
                continue
            phrases.append(t.lower())
    return phrases


def message_is_whitelisted(msg, whitelist):
    m = (msg or '').lower()
    for ph in whitelist:
        if ph in m:
            return True
    return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', default='data/hard_ham.csv')
    p.add_argument('--whitelist', default='whitelist_phrases.txt')
    p.add_argument('--out', default='data/hard_ham_verified.csv')
    args = p.parse_args()

    inp = Path(args.input)
    if not inp.exists():
        raise SystemExit('Input not found: ' + str(inp))

    whitelist = load_whitelist(args.whitelist)

    rows = []
    with inp.open('r', encoding='utf8', errors='ignore') as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            msg = r.get('message') or ''
            prob = r.get('prob') or r.get('p') or ''
            rows.append({'message': msg, 'prob': prob})

    before = len(rows)
    kept = [r for r in rows if not message_is_whitelisted(r['message'], whitelist)]
    after = len(kept)

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with outp.open('w', encoding='utf8', newline='') as fh:
        writer = csv.writer(fh)
        writer.writerow(['message', 'prob'])
        for r in kept:
            writer.writerow([r['message'], r['prob']])

    # Also write a copy with a filename that matches reentreno glob
    norm = outp.parent / ('normalized_' + outp.name)
    with norm.open('w', encoding='utf8', newline='') as fh:
        writer = csv.writer(fh)
        writer.writerow(['label', 'message'])
        for r in kept:
            writer.writerow(['ham', r['message']])

    print(f'Filtered {before} -> {after} examples. Wrote {outp} and {norm}')


if __name__ == '__main__':
    main()
