Detector de Spam con LSTM (notebook)

Contenido:
- `spam_lstm_notebook.ipynb`: Notebook con descarga de dataset, preprocesamiento, entrenamiento y evaluación.
- `train.py`: script para entrenar desde terminal y guardar `spam_lstm_model.h5` y `tokenizer.pkl`.
- `requirements.txt`: dependencias.

Instrucciones rápidas:

1. Crear un entorno virtual (opcional) y activar (PowerShell):

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
```

2. Instalar dependencias:

```powershell
pip install -r requirements.txt
```

3. Ejecutar el notebook con Jupyter o entrenar desde terminal:

```powershell
python train.py
```

4. El modelo y tokenizer quedan en la carpeta del notebook: `spam_lstm_model.h5`, `tokenizer.pkl`.

Nota: el dataset utilizado por defecto es el UCI SMS Spam Collection (en inglés). Para soportar español, añadir datasets en español y volver a entrenar.
 
Resumen de lo ejecutado
- Se creó `train.py` que descarga el UCI SMS Spam Collection, entrena una red LSTM y guarda:
	- `spam_lstm_model.h5`
	- `tokenizer.pkl`
- Resultado del entrenamiento (ejemplo obtenido localmente): Accuracy ~0.866 en test set. Observación: el `classification_report` mostró `ham`/`spam` porque esos son los labels originales del dataset.

Mostrar tablas en español
- El reporte de sklearn usa los nombres de clase que le pases. Para mostrarlas en español, en la evaluación cambia la llamada a `classification_report` así:

```python
from sklearn.metrics import classification_report
print(classification_report(y_test, preds, target_names=['no deseado','spam']))
```

Soporte para mensajes en español
- Pasos recomendados para soportar español y mejorar precisión:
	1. Recolecta o descarga datasets en español (ej: corpus de correos en español, SMS en español o datasets de Kaggle). Colócalos en `lstm_notebook/data/`.
	2. Normaliza acentos y caracteres: usa `unicodedata.normalize` o `unidecode` para eliminar tildes antes de tokenizar.
	3. Ajusta `train.py` para cargar y concatenar múltiples CSV (u otros formatos). Ejemplo breve en `train.py`:

```python
import glob
import pandas as pd

files = glob.glob('data/*.csv')
dfs = []
for f in files:
		d = pd.read_csv(f)
		# Suponer columnas: 'label' y 'message'
		dfs.append(d[['label','message']])
df = pd.concat(dfs, ignore_index=True)
```

	4. Quitar stopwords en español (nltk o spaCy) antes de entrenar.
	5. Experimentar con embeddings preentrenados en español (FastText, GloVe-es) o usar `Embedding layer` entrenable.

Cómo reentrenar con `py -3` (PowerShell)

```powershell
cd lstm_notebook
py -3 -m venv .venv
.\.venv\Scripts\pip.exe install -r .\requirements.txt
.\.venv\Scripts\python.exe train.py
```

Cómo integrar la versión web (ya incluida como demo)
- Ejecuta `train.py` para generar `spam_lstm_model.h5` y `tokenizer.pkl`.
- Desde `web_demo/` instala dependencias y ejecuta `app.py` (Flask). La app espera encontrar los archivos en `..\lstm_notebook\`.

Siguientes pasos sugeridos
- Añadir datasets en español en `lstm_notebook/data/` y reentrenar.
- Añadir normalización de acentos y stopwords en `train.py`.
- Ejecutar experimentos comparando `Embedding layer` vs `FastText` y registrar métricas (precision/recall especialmente para la clase `spam`).
- Generar un pequeño informe (Word/PDF) con resultados, discusión de impacto y mejoras propuestas.
