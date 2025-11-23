# lstm_notebook/convert_smsspam.py
"""Convertir SMSSpamCollection (tab-separated, no header) a normalized CSV
"""
import os
in_path = os.path.join(os.path.dirname(__file__), 'data', 'SMSSpamCollection.csv')
out_path = os.path.join(os.path.dirname(__file__), 'data', 'normalized_SMSSpamCollection.csv')

if not os.path.exists(in_path):
    print('No existe', in_path)
else:
    with open(in_path, 'rb') as f:
        raw = f.read()
    # intentar decodificar en utf-8 o latin1
    for enc in ('utf-8', 'latin-1', 'iso-8859-1'):
        try:
            text = raw.decode(enc)
            used_enc = enc
            break
        except Exception:
            text = None
    if text is None:
        print('No pude decodificar el archivo con utf-8/latin1')
    else:
        lines = text.splitlines()
        out_lines = ['label,message']
        for ln in lines:
            if not ln.strip():
                continue
            # split at first tab or multiple whitespace
            if '\t' in ln:
                parts = ln.split('\t', 1)
            else:
                parts = ln.split(None, 1)
            if len(parts) == 2:
                lab = parts[0].strip()
                msg = parts[1].strip().replace('\r','').replace('\n',' ')
                # escape double quotes
                msg = msg.replace('"','""')
                out_lines.append(f'{lab},"{msg}"')
        with open(out_path, 'w', encoding='utf-8') as of:
            of.write('\n'.join(out_lines))
        print('Convertido', in_path, '->', out_path, ' (encoding usado:', used_enc,')')
