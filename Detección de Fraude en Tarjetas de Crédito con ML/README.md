# 🔍 Detección de Fraude en Tarjetas de Crédito (ML)

Modelos de Machine Learning para detectar transacciones fraudulentas en un dataset **altamente
desbalanceado** (el fraude representa < 1 % de los casos).

## 🧪 Enfoque
- EDA, escalado y manejo del desbalance con **SMOTE**.
- Comparación de **5 modelos**: Random Forest, AdaBoost, XGBoost, LightGBM y LightGBM (K-Fold CV).
- Métricas adecuadas a clases desbalanceadas: **ROC-AUC, precision-recall, F1**.

## 📊 Resultados
- Mejor modelo: **LightGBM, ROC-AUC 0,980**.
- Random Forest: ROC-AUC 0,975 (F1 0,70).

## 📦 Datos
Dataset **Credit Card Fraud Detection** (Kaggle, MLG-ULB) — `creditcard.csv` (~144 MB).
**No se incluye** en el repo porque supera el límite de 100 MB de GitHub. Descárgalo de
<https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud> y colócalo en esta carpeta.

## 🛠️ Stack
Python · scikit-learn · XGBoost · LightGBM · imbalanced-learn (SMOTE) · Pandas · Matplotlib
