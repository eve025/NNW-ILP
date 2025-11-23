# lstm_notebook/normalize_data.py
"""Leer cada CSV en data/, detectar separador/encoding, limpiar headers y guardar copias normalizadas.
Salida: data/normalized_<orig>.csv con columnas `label,message`.
"""
import os, glob
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')

COMMON_ENCODINGS = ['utf-8', 'latin-1', 'iso-8859-1']


def looks_like_label(series):
    vals = series.dropna().astype(str).str.lower().str.strip()
    unique_vals = set(vals.unique())
    common_labels = {'spam','ham','s','h','0','1','true','false','yes','no','tipo'}
    if len(unique_vals) <= 20 and len(unique_vals & common_labels) > 0:
        return True
    if len(unique_vals) <= 10 and vals.str.len().mean() < 6:
        return True
    return False


def try_read(path):
    # try several encodings and let pandas infer separator
    for enc in COMMON_ENCODINGS:
        try:
            df = pd.read_csv(path, sep=None, engine='python', encoding=enc)
            return df, enc
        except Exception:
            continue
    # last resort: try reading without separator inference
    for enc in COMMON_ENCODINGS:
        try:
            df = pd.read_csv(path, encoding=enc)
            return df, enc
        except Exception:
            continue
    return None, None


def normalize_file(path):
    print('-'*60)
    print('Procesando:', path)
    df, enc = try_read(path)
    if df is None:
        print('  ERROR: no pude leer el archivo con los encodings probados.')
        return
    print('  Leído con encoding:', enc)
    # strip column names
    df.columns = [str(c).strip() for c in df.columns]
    cols = list(df.columns)
    print('  Columnas detectadas:', cols)
    if len(cols) < 2:
        print('  Menos de 2 columnas — omitiendo.')
        return
    # if many columns, try to find label/message by column name
    label_col = None
    message_col = None
    if len(cols) > 2:
        name_candidates_label = ['label', 'tipo', 'category', 'clasificacion', 'etiqueta', 'categoria']
        name_candidates_message = ['message', 'mensaje', 'cuerpo', 'body', 'text', 'contenido']
        for c in cols:
            low = c.lower()
            for cand in name_candidates_label:
                if cand in low:
                    label_col = c
                    break
            if label_col:
                break
        for c in cols:
            low = c.lower()
            for cand in name_candidates_message:
                if cand in low:
                    message_col = c
                    break
            if message_col:
                break
        if label_col and message_col:
            tmp = df[[label_col, message_col]].copy()
            print(f'  Usando columnas detectadas -> label: {label_col}, message: {message_col}')
        else:
            # fallback to first two
            tmp = df.iloc[:, :2].copy()
            print('  No se detectaron columnas explícitas de label/message; usaré las dos primeras columnas como antes.')
    else:
        # take first two columns
        tmp = df.iloc[:, :2].copy()
    c0, c1 = tmp.columns[0], tmp.columns[1]
    if looks_like_label(tmp[c0]) and not looks_like_label(tmp[c1]):
        tmp.columns = ['label','message']
    elif looks_like_label(tmp[c1]) and not looks_like_label(tmp[c0]):
        tmp = tmp[[c1, c0]].copy()
        tmp.columns = ['label','message']
    else:
        # fallback: assume second column is message if longer
        len0 = tmp[c0].astype(str).str.len().median()
        len1 = tmp[c1].astype(str).str.len().median()
        if len0 < len1:
            tmp.columns = ['label','message']
        else:
            tmp = tmp[[c1, c0]].copy()
            tmp.columns = ['label','message']
    # normalize label values
    tmp['label'] = tmp['label'].astype(str).str.lower().str.strip()
    tmp['message'] = tmp['message'].astype(str)
    # save normalized copy
    base = os.path.basename(path)
    out = os.path.join(DATA_DIR, f'normalized_{base}')
    tmp.to_csv(out, index=False, encoding='utf-8')
    print('  Guardado normalizado ->', out)
    # brief stats
    print('  Labels unique (sample):', list(tmp['label'].unique())[:20])
    print('  Counts:')
    print(tmp['label'].value_counts().head(20))


if __name__ == '__main__':
    files = glob.glob(os.path.join(DATA_DIR, '*.csv'))
    if not files:
        print('No se encontraron CSVs en', DATA_DIR)
    for f in files:
        normalize_file(f)
    print('\nHecho. Revisa los archivos `normalized_*.csv` y, si todo está correcto, re-entrena con ellos (o reemplaza los originales tras hacer backup).')
