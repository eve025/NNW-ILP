AgenteDeIA — Detector LSTM de Spam (notebook + demo web)

Instrucciones rápidas para poner en marcha el proyecto en Windows (PowerShell).

Requisitos
- Python 3.8+ instalado y disponible como `python` o `py -3`.
- Windows PowerShell (v5.1) o superior.

Pasos automáticos (recomendado)
1. Abre PowerShell en la raíz del repositorio.
2. Permite ejecución temporal de scripts (si no lo has hecho):
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
```
3. Ejecuta el script de setup (crea `.venv`, instala TensorFlow y paquetes):
```powershell
.\setup_env.ps1
```

Esto creará `.venv` en la raíz, instalará TensorFlow y los requisitos listados en `lstm_notebook/requirements.txt` y `web_demo/requirements.txt`.

Activar el entorno manualmente
```powershell
.\.venv\Scripts\Activate.ps1

# comprobar instalación
python -c "import sys, tensorflow as tf; print('PYTHON:', sys.executable); print('TF version:', tf.__version__)"
```

# AgenteDeIA — Detector LSTM de Spam (notebook + demo web)

Este repositorio contiene código para entrenar y evaluar un detector LSTM de spam (`lstm_notebook`), utilidades para mejorar la calidad del modelo y una demo web.

## Qué dejar en el README (resumen para colaboradores)
- Requisitos y creación del entorno virtual (`.venv`).
- Comandos para entrenar: `train.py` (entrenamiento desde cero) y `retrain_with_hard_negatives.py` (reentrenos/fine-tune con ejemplos difíciles).
- Cómo ejecutar la demo web (`app.py` o `server.js` según el subproyecto web).
- Utilidades principales: `collect_hard_ham.py`, `improve_tokenizer.py`, `calibrate_and_select_threshold.py`, `hybrid_policy.py`.
- Notas sobre protobuf y cómo mitigar las advertencias (ver `lstm_notebook/PROTOBUF_UPDATE.md` y `lstm_notebook/PROTOBUF_UPDATE_ES.md`).

## Requisitos
- Python 3.8+ (recomendado), accesible como `py -3`.
- PowerShell en Windows (v5.1) o similar.

## Instalación recomendada (PowerShell)
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
py -3 -m pip install --upgrade pip
py -3 -m pip install -r .\lstm_notebook\requirements.txt
```

## Activar el entorno
```powershell
.\.venv\Scripts\Activate.ps1
python -c "import sys, tensorflow as tf; print('PYTHON:', sys.executable); print('TF version:', tf.__version__)"
```

## Entrenamiento

- Entrenamiento completo desde cero (genera artefactos en `lstm_notebook/`):
```powershell
# dentro del venv activado
py -3 .\lstm_notebook\train.py
```

- Reentreno / fine-tune incorporando ham difíciles (`hard_ham.csv`) y usando `class_weight='balanced'`:
```powershell
.\.venv\Scripts\Activate.ps1
$env:EPOCHS=10
$env:BATCH_SIZE=64
py -3 .\lstm_notebook\retrain_with_hard_negatives.py
```

> Nota: proponemos `EPOCHS=10` para un reentreno de validación; ajusta según tiempo y recursos.

## Inferencia / Demo Web

- Demo Python/Flask (si existe `web_demo/app.py` o `app.py`):
```powershell
# con el venv activado
py -3 .\web_demo\app.py
```

- Si la demo es Node.js / `server.js`:
```powershell
npm install
node server.js
```

## Flujo recomendado para mejorar precisión operativa
1. Generar o recolectar ham difíciles (ej.: `lstm_notebook/data/normalized_extra.csv`):
  ```powershell
  py -3 .\lstm_notebook\collect_hard_ham.py --inputs .\lstm_notebook\data\normalized_extra.csv --threshold 0.42
  ```
  Resultado: `lstm_notebook/data/hard_ham.csv` (usar para reentreno).

2. Construir/actualizar tokenizer robusto:
  ```powershell
  py -3 .\lstm_notebook\improve_tokenizer.py --max_words 20000
  ```
  Guarda `lstm_notebook/tokenizer.pkl`.

3. Reentrenar con `hard_ham.csv` (fine-tune): ver sección Entrenamiento.

4. Calibrar probabilidades y seleccionar umbrales por idioma:
  ```powershell
  py -3 .\lstm_notebook\calibrate_and_select_threshold.py --model .\lstm_notebook\spam_lstm_model_refit.h5 --tokenizer .\lstm_notebook\tokenizer_refit.pkl --lang all
  ```
  Salidas: `calibrator_<lang>.pkl`, `retrain_by_lang_thresholds_<lang>_calibrated.csv`, `recommended_threshold_<lang>.txt`.

5. Política híbrida en producción (modelo + reglas + whitelist):
  ```powershell
  py -3 .\lstm_notebook\hybrid_policy.py --message "Texto a clasificar" --threshold 0.42
  ```

## Notas sobre protobuf (warnings)

Si ves warnings como "Protobuf gencode version ... older than runtime", sigue `lstm_notebook/PROTOBUF_UPDATE_ES.md`.

## Archivos importantes
- `lstm_notebook/train.py` — entrenamiento base.
- `lstm_notebook/retrain_with_hard_negatives.py` — reentrenos con hard negatives.
- `lstm_notebook/tokenizer.pkl` / `tokenizer_refit.pkl` — tokenizadores.
- `lstm_notebook/spam_lstm_model.h5` / `spam_lstm_model_refit.h5` — modelos.
- `lstm_notebook/hybrid_policy.py` — política de inferencia (whitelist + heurísticas).

---
## Cómo ejecutar la libreta (`lstm_notebook`) y la interfaz estática o demo web

1) Ejecutar la libreta (`lstm_notebook/spam_lstm_notebook.ipynb`) en Windows (PowerShell):

```powershell
cd lstm_notebook
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r ..\lstm_notebook\requirements.txt  # si el archivo existe; de lo contrario instala packages mínimos
jupyter notebook
# Abrir en el navegador la libreta `spam_lstm_notebook.ipynb` y ejecutar las celdas (Run -> Run All)
```

Las celdas nuevas A5/A6/A7 incluyen: definición del proyecto, descripción de los datos y un experimento demo que guarda `spam_lstm_model_demo.h5`, `tokenizer_demo.pkl` y `experiment_report.txt` en la carpeta `lstm_notebook`.

2) Abrir `index.html` (interfaz estática) desde la raíz del repo (método recomendado: servidor simple):

```powershell
cd ..\  # vuelve a la raíz si estás en lstm_notebook
py -3 -m http.server 8000
# Abrir en el navegador: http://127.0.0.1:8000/index.html
```

3) Levantar la demo Flask (backend) desde `web_demo`:

```powershell
cd web_demo
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
py -3 app.py
# Abrir en el navegador: http://127.0.0.1:5000/
```

4) Evidencia de experimentación (video ≤ 5 minutos):


Si quieres, puedo preparar un pequeño script para automatizar la generación del `experiment_report.txt` (ya incluido en la celda A7) y recomendaciones para la grabación.

## A7 - Entregable y notebook específico

He añadido `lstm_notebook/a7_notebook.ipynb` para comprobar rápidamente un modelo entrenado y generar predicciones de ejemplo.

Qué anexar para la entrega A7:
- `lstm_notebook/artifacts/spam_lstm_model_demo.h5` o `spam_lstm_model_refit.h5`
- `lstm_notebook/artifacts/tokenizer_demo.pkl` o `tokenizer_refit.pkl`
- `lstm_notebook/artifacts/experiment_report.txt`

El notebook A7 carga el `.h5` y el `.pkl`, muestra el `model.summary()`, realiza predicciones de ejemplo y describe brevemente el impacto social/tecnológico y recomendaciones para mejorar el desempeño (más datos, embeddings preentrenados, balanceo, regularización).

Para cualquier modificación adicional o para que haga el commit final con firma, dime y lo hago.

