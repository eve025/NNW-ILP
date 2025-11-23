# lstm_notebook/inspect_data.py
"""Inspecciona CSVs en data/ y muestra: columnas, primeras filas, conteo de etiquetas
Heurísticas para detectar qué columna es label y cuál es message.
"""
import os, glob
import pandas as pd

def looks_like_label(series):
    vals = series.dropna().astype(str).str.lower().str.strip()
    unique_vals = set(vals.unique())
    common_labels = {'spam','ham','s','h','0','1','true','false','yes','no','tipo'}
    if len(unique_vals) <= 20 and len(unique_vals & common_labels) > 0:
        return True
    if len(unique_vals) <= 10 and vals.str.len().mean() < 6:
        return True
    return False


def inspect_file(path):
    print('='*80)
    print('FILE:', path)
    try:
        df = pd.read_csv(path, nrows=50)
    except Exception as e:
        print('  ERROR reading file:', e)
        return
    cols = list(df.columns)
    print('  Columns:', cols)
    print('  Sample rows (up to 5):')
    print(df.head(5).to_string(index=False))

    # if there are at least 2 columns, test heuristics
    if len(cols) >= 2:
        c0 = df.iloc[:,0]
        c1 = df.iloc[:,1]
        print('  Heurística label?', cols[0], looks_like_label(c0))
        print('  Heurística label?', cols[1], looks_like_label(c1))
        # counts if detected
        if looks_like_label(c0) and not looks_like_label(c1):
            lab = c0.astype(str).str.strip()
            print('  Detected label column:', cols[0])
            print('   Unique labels (sample):', lab.unique()[:20])
            print('   Value counts (top):')
            print(lab.value_counts().head(20))
        elif looks_like_label(c1) and not looks_like_label(c0):
            lab = c1.astype(str).str.strip()
            print('  Detected label column:', cols[1])
            print('   Unique labels (sample):', lab.unique()[:20])
            print('   Value counts (top):')
            print(lab.value_counts().head(20))
        else:
            print('  No quedó claro qué columna es label. Revisar manualmente.')
    else:
        print('  Menos de 2 columnas; no usable como dataset de mensajes.')


if __name__ == '__main__':
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    files = glob.glob(os.path.join(data_dir, '*.csv'))
    if not files:
        print('No se encontraron CSVs en', data_dir)
    for f in files:
        inspect_file(f)
    print('\nRecomendación: asegúrate de que cada CSV tenga columnas `label` y `message` (o la primera sea label y la segunda message).')
    print('Si tus cabeceras están en español, por ejemplo `tipo,mensaje`, renómbralas o usa el script para preprocesar.')
