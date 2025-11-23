import os
import pandas as pd

p = os.path.join(os.path.dirname(__file__), 'data', 'correos.csv')
out = os.path.join(os.path.dirname(__file__), 'data', 'normalized_correos_fixed.csv')

# detectar encoding
df = None
for enc in ('utf-8', 'latin-1', 'iso-8859-1'):
    try:
        df = pd.read_csv(p, encoding=enc)
        used_enc = enc
        break
    except Exception:
        df = None
else:
    raise RuntimeError('No pude leer correos.csv con los encodings probados')

print('Leído correos.csv con encoding:', used_enc)
print('Columnas:', list(df.columns))

# columnas candidatas
label_candidates = [c for c in df.columns if c.lower() in ('etiqueta','clasificacion','label','tipo','categoria')]
message_candidates = [c for c in df.columns if c.lower() in ('cuerpo','asunto','mensaje','body','message','texto')]

print('Label candidates:', label_candidates)
print('Message candidates:', message_candidates)

# elegir mejor columna de etiqueta por número de non-null y pocas categorías
best_label = None
best_nonnull = -1
for c in label_candidates:
    nonnull = df[c].dropna().shape[0]
    unique_vals = set(df[c].dropna().astype(str).str.lower().str.strip().unique())
    # penalizar si demasiadas categorías
    score = nonnull - len(unique_vals)
    print(f'  {c}: non-null={nonnull}, unique={len(unique_vals)}, score={score}')
    if nonnull > best_nonnull:
        best_nonnull = nonnull
        best_label = c

# fallback: si no hay candidatas, buscar la columna con más non-null
if best_label is None:
    for c in df.columns:
        nonnull = df[c].dropna().shape[0]
        if nonnull > best_nonnull:
            best_nonnull = nonnull
            best_label = c

print('Selected label column:', best_label)

# elegir columna de mensaje (preferir cuerpo/mensaje/asunto)
best_msg = None
for pref in ('cuerpo','mensaje','asunto','message','body','texto'):
    for c in df.columns:
        if pref == c.lower():
            best_msg = c
            break
    if best_msg:
        break
if best_msg is None:
    # fallback a la columna con texto más largo median
    best_len = -1
    for c in df.columns:
        try:
            avg_len = df[c].astype(str).str.len().median()
        except Exception:
            avg_len = 0
        if avg_len > best_len:
            best_len = avg_len
            best_msg = c

print('Selected message column:', best_msg)

# construir columna label normalizada
lab = df[best_label].astype(str).str.lower().str.strip()
# mapear posibles variantes a 'spam'/'ham'
lab = lab.replace({'spam':'spam','ham':'ham','s':'spam','h':'ham','spam ':'spam',' ham':'ham'})
# si contiene la palabra spam/ham dentro del string, extraerla
lab = lab.apply(lambda x: 'spam' if isinstance(x,str) and 'spam' in x.lower() else ('ham' if isinstance(x,str) and 'ham' in x.lower() else x))

# si hay columna 'clasificacion' y lab es no válido, usarla como respaldo
if 'clasificacion' in df.columns:
    cls = df['clasificacion'].astype(str).str.lower().str.strip()
    lab = lab.where(lab.notna() & (lab!='nan'), cls)

# limpiar 'nan' y filas sin label
lab = lab.replace({'nan': None})
mask_valid = lab.notna()
print('Total rows:', len(df), 'valid label rows after merge:', mask_valid.sum())

# preparar mensaje
msg = df[best_msg].astype(str)
res = pd.DataFrame({'label': lab, 'message': msg})
res = res[mask_valid]
# guardar
res.to_csv(out, index=False, encoding='utf-8')
print('Guardado fichero corregido en:', out)