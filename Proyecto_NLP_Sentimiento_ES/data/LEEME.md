# Carpeta de datos

Este proyecto usa un dataset de **~2.590 tuits en español** etiquetados con emociones
(columnas `text` y `sentiment`). Por buenas prácticas, **los CSV no se versionan** (.gitignore).

## Ejecutar con datos reales
Coloca tu archivo como **`sentiment_analysis_dataset.csv`** (columnas `text`, `sentiment`).
El notebook lo detecta, mapea las 6 emociones a binario
(positivo: peaceful/powerful/joyful · negativo: mad/sad/scared) y entrena.

## Alternativa automática (otro dataset en español)
```bash
pip install datasets pandas
python data/descargar_dataset.py     # descarga 'muchocine' (reseñas de cine en español)
```

## Si no hay datos
El notebook genera un pequeño **corpus de demostración** para ejecutarse igual.
