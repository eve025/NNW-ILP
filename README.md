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

Entrenar el modelo (opcional)
```powershell
# dentro del venv activado
python .\lstm_notebook\train.py
```
Esto generará `lstm_notebook\spam_lstm_model.h5`, `tokenizer.pkl`, `training_plot.png`, `training_report.txt` y `best_threshold.txt`.

Arrancar la demo web
```powershell
# con el venv activado
python .\web_demo\app.py
```
Abrir `http://127.0.0.1:5000/` en el navegador.

Diagnóstico y utilidades
- `web_demo\quick_test.py`: ejemplo rápido de inferencia en consola.
- `web_demo\check_model_weights.py`: inspecciona la última capa y hace pruebas controladas.
- Si VS Code muestra "reportMissingImports" para `tensorflow.keras`, selecciona el intérprete del venv:
  - Ctrl+Shift+P → "Python: Select Interpreter" → elige `.venv\Scripts\python.exe`.

Notas
- El archivo `.vscode/settings.json` puede apuntar al intérprete `.venv` para ayudar a colaboradores que usen VS Code.
- El script `setup_env.ps1` usa `python` o `py -3` para crear el venv; si tu instalación de Python usa otra ruta, activa manualmente el entorno y ejecuta los comandos `python -m pip install ...` mostrados en el script.

Si quieres, puedo añadir un endpoint `/status` en la demo para mostrar si modelo/tokenizer/umbral están cargados, o insertar los formularios de prueba directamente en `web_demo/templates/index.html`.
