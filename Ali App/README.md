# 🪙 APP ALI — Detección y Autenticación de Billetes Bolivianos

Sistema de visión por computadora que identifica la **denominación** de billetes bolivianos y
verifica su **autenticidad**, para asistir a **personas con discapacidad visual** mediante audio
y vibración. Proyecto de grado · UNIFRANZ · 2025–2026.

## 📊 Resultados
- Denominación (5 clases): **98,35 % de accuracy**.
- Autenticidad (auténtico / falso): **AUC 0,885**.
- Comparación de 3 enfoques de detección de anomalías: CNN supervisado, **RBM** (AUC 0,776) y **SOM** (AUC 0,885).

## 🧩 Pipeline (scripts)
- `00_limpiar_dataset.py` — limpieza del dataset (orientación EXIF, borrosas, duplicados).
- `06_entrenar_modelo.py` — entrenamiento (MobileNetV2 + Transfer Learning + Fine-Tuning).
- `07_evaluar_modelo.py` — evaluación del modelo.
- `17_rbm_autenticidad.py` — detección de anomalías con RBM (Boltzmann).
- `18_som_autenticidad.py` — detección de anomalías con SOM (Kohonen).

## 🛠️ Stack
Python · TensorFlow · Keras · TFLite · scikit-learn · OpenCV · (app móvil en Flutter)

> ⚠️ El **dataset de billetes no se incluye** (privado y sensible). Los modelos se exportan a
> TFLite e integran en una app Flutter con texto a voz y retroalimentación háptica.
