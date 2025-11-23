# lstm_notebook/check_correos_labels.py
import pandas as pd
import os
p = os.path.join(os.path.dirname(__file__), 'data', 'correos.csv')
print('File:', p)
for enc in ('utf-8','latin-1','iso-8859-1'):
    try:
        df = pd.read_csv(p, encoding=enc)
        print('Read with encoding:', enc)
        break
    except Exception as e:
        print('Failed with', enc, '->', e)
else:
    raise RuntimeError('Could not read correos.csv')
print('Columns:', list(df.columns))
for col in ['etiqueta','clasificacion','cuerpo','asunto']:
    if col in df.columns:
        s = df[col]
        nonnull = s.dropna().shape[0]
        print(f"{col}: non-null={nonnull}, unique_sample={list(s.dropna().astype(str).unique()[:10])}")

# show first 10 rows for manual inspection
print('\nFirst 10 rows:')
print(df.head(10).to_string(index=False))
