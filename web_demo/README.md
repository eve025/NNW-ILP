Demo web para el detector de spam

Instrucciones rápidas (PowerShell):

1. Crear y activar entorno virtual:

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
```

2. Instalar dependencias:

```powershell
pip install -r requirements.txt
```

3. Asegurarse de que `spam_lstm_model.h5` y `tokenizer.pkl` estén en `../lstm_notebook/` (desde `web_demo` carpeta). Puedes generar esos archivos ejecutando `lstm_notebook/train.py`.

4. Ejecutar demo:

```powershell
python app.py
```

5. Abrir en el navegador: `http://127.0.0.1:5000/`.

Ver reporte y gráfico de entrenamiento desde la web
- Si entrenaste con `train.py`, el script genera en `lstm_notebook/` los archivos:
	- `training_report.txt` (métricas: classification_report, confusion_matrix, accuracy)
	- `training_plot.png` (curvas de loss y accuracy)

- Rutas disponibles en la demo web:
	- `http://127.0.0.1:5000/report` → descarga/visualiza `training_report.txt`.
	- `http://127.0.0.1:5000/plot` → muestra la imagen `training_plot.png`.

Nota: asegúrate de ejecutar `train.py` antes de abrir `/report` o `/plot`.
Notas sobre soporte en español y despliegue
- La demo web usa el modelo guardado en `..\lstm_notebook\spam_lstm_model.h5` y `..\lstm_notebook\tokenizer.pkl`.
- Para que el detector soporte correctamente mensajes en español, reentrena el modelo en `lstm_notebook` con datasets en español y/o normaliza acentos antes de tokenizar (ver `lstm_notebook/README.md` para pasos concretos).

Despliegue y producción
- Para exponer la app en Internet (uso académico o demo), considera:
	- Desplegar en un PaaS (Render, Railway, Heroku) o usar contenedor Docker.
	- Proteger la app con autenticación si se va a usar en un entorno institucional.
	- Usar HTTPS y validar entradas (evitar inyección de comandos en logs).

Prueba rápida local
1. Genera `spam_lstm_model.h5` y `tokenizer.pkl` corriendo `train.py`.
2. Desde `web_demo` ejecutar:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\pip.exe install -r .\requirements.txt
.\.venv\Scripts\python.exe app.py
```

3. Abrir `http://127.0.0.1:5000/` y probar mensajes en español tras reentrenar.
