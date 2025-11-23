#!/usr/bin/env python3
"""ingest_extra_ham.py

Lee uno o varios CSVs con columnas `label,message`, limpia y deduplica
y escribe `lstm_notebook/data/normalized_extra.csv` listo para reentreno.

Uso:
  - sin argumentos intentará usar el archivo en Downloads: ~/Downloads/train_ham_50.csv
  - o pasa rutas de entrada: python ingest_extra_ham.py path1.csv path2.csv

El script:
  - unifica nombres de columnas (si es necesario)
  - limpia espacios y comillas
  - elimina mensajes vacíos
  - elimina duplicados (por texto y etiqueta)
  - guarda en `data/normalized_extra.csv`
"""
import os
import sys
import io
import argparse
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE, 'data')
OUT_PATH = os.path.join(DATA_DIR, 'normalized_extra.csv')


def guess_default_paths():
    # Prefer the project's data folder first, then Downloads
    project_path = os.path.join(BASE, 'data', 'train_ham_50.csv')
    if os.path.exists(project_path):
        return [project_path]
    up = os.environ.get('USERPROFILE') or os.path.expanduser('~')
    downloads_path = os.path.join(up, 'Downloads', 'train_ham_50.csv')
    return [downloads_path]


def read_input_paths(paths):
    frames = []
    for p in paths:
        if not os.path.exists(p):
            print(f'Warning: input not found: {p}', file=sys.stderr)
            continue
        try:
            df = pd.read_csv(p)
        except Exception:
            # fallback: try a robust manual parser that splits only on the first comma
            try:
                rows = []
                with open(p, 'r', encoding='utf8', errors='replace') as fh:
                    for i, ln in enumerate(fh):
                        ln = ln.strip('\n')
                        if not ln:
                            continue
                        # skip header if present
                        if i == 0 and ('label' in ln.lower() and 'message' in ln.lower()):
                            continue
                        # try split at first comma (label,message)
                        if ',' in ln:
                            a, b = ln.split(',', 1)
                        else:
                            parts = ln.split('\t', 1)
                            if len(parts) >= 2:
                                a, b = parts[0], parts[1]
                            else:
                                # cannot parse this line reliably
                                continue
                        a = a.strip().strip('"\'')
                        b = b.strip().strip('"\'')
                        rows.append((a, b))
                if not rows:
                    raise ValueError('manual parse yielded no rows')
                df = pd.DataFrame(rows, columns=['label', 'message'])
            except Exception as e:
                print(f'Failed to read {p}: {e}', file=sys.stderr)
                continue

        # normalize first two columns to label,message if necessary
        cols = [c.lower().strip() for c in df.columns]
        if 'label' in cols and 'message' in cols:
            # rename to standard if different cases
            col_map = {df.columns[i]: name for i, name in enumerate(cols) if name in ('label','message')}
            df = df.rename(columns=col_map)
            df = df[['label','message']]
        else:
            # take first two columns
            df = df.iloc[:, :2]
            df.columns = ['label','message']

        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=['label','message'])
    return pd.concat(frames, ignore_index=True)


def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    # Strip and normalize
    df['message'] = df['message'].astype(str).str.strip()
    df['label'] = df['label'].astype(str).str.strip().str.lower()
    # remove empty messages
    df = df[df['message'].str.len() > 0].copy()
    # collapse multiple spaces
    df['message'] = df['message'].str.replace(r'\s+', ' ', regex=True)
    # remove surrounding quotes
    df['message'] = df['message'].str.strip('"\'')
    # drop exact duplicates (label + message)
    before = len(df)
    df = df.drop_duplicates(subset=['label','message']).reset_index(drop=True)
    after = len(df)
    print(f'Removed {before-after} duplicate rows')
    return df


def ensure_data_dir():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR, exist_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description='Ingest extra ham CSV(s) into data/normalized_extra.csv')
    parser.add_argument('inputs', nargs='*', help='CSV input paths (label,message). If empty, tries Downloads/train_ham_50.csv')
    parser.add_argument('--append', action='store_true', help='Append to existing normalized_extra.csv instead of overwriting')
    args = parser.parse_args(argv)

    inputs = args.inputs or guess_default_paths()
    print('Inputs:', inputs)
    df = read_input_paths(inputs)
    if df.empty:
        print('No valid input files found or they were empty. Exiting.', file=sys.stderr)
        sys.exit(1)

    df = clean_df(df)
    ensure_data_dir()

    if args.append and os.path.exists(OUT_PATH):
        existing = pd.read_csv(OUT_PATH)
        combined = pd.concat([existing, df], ignore_index=True)
        combined = combined.drop_duplicates(subset=['label','message']).reset_index(drop=True)
        combined.to_csv(OUT_PATH, index=False)
        print(f'Appended and wrote {len(combined)} rows to {OUT_PATH}')
    else:
        df.to_csv(OUT_PATH, index=False)
        print(f'Wrote {len(df)} rows to {OUT_PATH}')


if __name__ == '__main__':
    main()
