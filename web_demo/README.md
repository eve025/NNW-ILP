Demo web para el detector de spam

Resumen rápido
La demo Flask muestra una UI donde puedes pegar un mensaje y obtener una predicción ("Spam" / "Legítimo"). Para que la demo cargue correctamente el modelo, debe haber artefactos generados por el proceso de entrenamiento en la carpeta `lstm_notebook/`.

Qué archivos necesita la demo
- `lstm_notebook/spam_lstm_model_refit.h5` (preferido) o `lstm_notebook/spam_lstm_model.h5` (fallback)
- `lstm_notebook/spam_lstm_model_refit.weights.h5` (usa si el `.h5` tiene capas custom y no se puede deserializar)
- `lstm_notebook/tokenizer_refit.pkl` (preferido) o `lstm_notebook/tokenizer.pkl` (fallback)
- `lstm_notebook/best_threshold_refit.txt` (preferido) o `lstm_notebook/best_threshold.txt` (fallback)

La app intenta cargar los artefactos "refit" primero; si `load_model()` falla (p. ej. por una capa custom), intenta reconstruir la arquitectura con `retrain_with_hard_negatives.build_model()` y cargar `*.weights.h5`.

Pasos para generar los artefactos (PowerShell)
1) (Opcional) revisar/corregir datos y generar `hard_ham_verified` si usas ejemplos recogidos manualmente:
```powershell
cd ..\lstm_notebook
# genera hard_ham desde archivos de datos (ejemplo ya incluido)
py -3 collect_hard_ham.py --inputs .\data\normalized_extra.csv .\data\normalized_correos.csv --model .\spam_lstm_model_refit.h5 --tokenizer .\tokenizer_refit.pkl --threshold 0.42
# filtra con whitelist (crea normalized_hard_ham_verified.csv)
py -3 verify_hard_ham.py --input .\data\hard_ham.csv --whitelist ..\lstm_notebook\whitelist_phrases.txt --out .\data\hard_ham_verified.csv
```

2) Reentrenar (produce `spam_lstm_model_refit.h5`, `spam_lstm_model_refit.weights.h5`, `tokenizer_refit.pkl` y `best_threshold_refit.txt`):
```powershell
cd ..\lstm_notebook
$env:EPOCHS=10; $env:BATCH_SIZE=64; py -3 retrain_with_hard_negatives.py
```

3) Verifica que los archivos se hayan creado en `lstm_notebook/`:
```powershell
ls .\spam_lstm_model_refit.* .\spam_lstm_model_refit.weights.h5 .\tokenizer_refit.pkl .\best_threshold_refit.txt
```

Arrancar la demo web (PowerShell)
1. Ir a la carpeta `web_demo` y (opcional) activar entorno virtual:
```powershell
cd web_demo
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```
2. Ejecutar la app:
```powershell
py -3 app.py
```
3. Abrir en el navegador: `http://127.0.0.1:5000/` (o `http://localhost:5000`).

Qué significa el mensaje "Error: Modelo no cargado. Entrena y copia los archivos."
- La aplicación no encontró los artefactos esperados en `../lstm_notebook/` o falló al cargarlos.
- Soluciones:
  - Ejecutar el reentreno usando `retrain_with_hard_negatives.py` (comando arriba).
  - Confirmar que `tokenizer_refit.pkl` / `tokenizer.pkl` y `spam_lstm_model_refit.h5` (o `spam_lstm_model.h5`) existen en `lstm_notebook/`.
  - Si el `.h5` no se carga por error de capa custom, asegúrate de que exista `spam_lstm_model_refit.weights.h5` en la misma carpeta; la app intentará reconstruir la arquitectura y cargar los pesos.

Comandos útiles de diagnóstico
```powershell
# listar archivos creados por el entrenamiento
ls ..\lstm_notebook\spam_lstm_model_refit.*
type ..\lstm_notebook\best_threshold_refit.txt
```

Notas adicionales
- Si la demo sigue mostrando el error tras generar los artefactos, pega aquí la salida del servidor (consola) y la mostraré la causa exacta y la solución.
- Para producción, guarda las versiones del modelo y del tokenizador junto con el código (tagging) y usar un formato Keras nativo `.keras` es recomendable (evita problemas con h5 legacy).

Soporte rápido
- Si quieres, arranco la demo ahora y comprobaré en la consola si el modelo se carga correctamente y te envío los pasos siguientes (o corrijo cualquier error de carga automático). Sólo dime si quieres que la inicie.
