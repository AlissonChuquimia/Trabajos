"""
=============================================================================
SCRIPT 07 - EVALUACION DETALLADA DEL MODELO
=============================================================================
QUE HACE ESTE SCRIPT:
  Carga el modelo entrenado y produce el reporte de evaluacion completo
  para incluir en el informe de tesis:

  1. Matriz de confusion 5x5 visualizada con seaborn
  2. Metricas por clase (precision, recall, F1)
  3. Promedios macro y weighted
  4. Curvas de entrenamiento (loss y accuracy por epoca)
  5. Reporte CSV de metricas por clase para auditar
  6. Reporte de texto plano legible para el informe

ENTRADA:
  models/mobilenet_billetes.keras
  models/mobilenet_billetes_history.json
  dataset_clasificacion/test/{clase}/*.jpg

SALIDA:
  reportes/07_matriz_confusion.png        (figura para el informe)
  reportes/07_curvas_entrenamiento.png    (figura para el informe)
  reportes/07_metricas_por_clase.csv      (tabla auditable)
  reportes/07_evaluacion_reporte.txt      (texto legible)

USO:
  .venv\\Scripts\\activate
  python scripts/07_evaluar_modelo.py                    # MobileNetV2 default
  python scripts/07_evaluar_modelo.py --arq efficient    # EfficientNet

REQUISITOS:
  tensorflow, scikit-learn, matplotlib, seaborn, pandas
=============================================================================
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_recall_fscore_support,
)


BASE_DIR     = Path(__file__).parent.parent
DATASET_DIR  = BASE_DIR / "dataset_clasificacion"
MODELS_DIR   = BASE_DIR / "models"
REPORTE_DIR  = BASE_DIR / "reportes"

# 5 clases de denominacion, orden numerico igual que script 06.
# IMPORTANTE: debe coincidir con _detectar_clases() de script 06.
CLASES = ["bs10", "bs20", "bs50", "bs100", "bs200"]


def cargar_test_set(img_size, batch_size):
    """Carga el conjunto de prueba sin shuffle (necesario para coordinar y_true / y_pred)."""
    return tf.keras.utils.image_dataset_from_directory(
        DATASET_DIR / "test",
        labels="inferred",
        label_mode="int",
        class_names=CLASES,
        color_mode="rgb",
        batch_size=batch_size,
        image_size=(img_size, img_size),
        shuffle=False,
    )


def predecir_todo(modelo, ds):
    """Devuelve y_true, y_pred (indices de clase) sobre el dataset."""
    y_true_all, y_pred_all = [], []
    for x_batch, y_batch in ds:
        probs = modelo.predict(x_batch, verbose=0)
        y_true_all.append(y_batch.numpy())
        y_pred_all.append(np.argmax(probs, axis=1))
    return np.concatenate(y_true_all), np.concatenate(y_pred_all)


def graficar_matriz_confusion(y_true, y_pred, clases, ruta):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(clases))))
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=clases, yticklabels=clases,
        cbar_kws={"label": "Cantidad de muestras"},
        ax=ax,
    )
    ax.set_xlabel("Clase predicha", fontsize=11)
    ax.set_ylabel("Clase real", fontsize=11)
    ax.set_title("Matriz de confusión sobre el conjunto de prueba", fontsize=13, pad=14)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    plt.setp(ax.get_yticklabels(), rotation=0)
    plt.tight_layout()
    plt.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close()
    return cm


def graficar_curvas(history_data, ruta):
    """Grafica loss y accuracy de las dos fases (TL + FT)."""
    tl = history_data.get("fase_tl", {})
    ft = history_data.get("fase_ft", {})

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # ----- accuracy -----
    ax = axes[0]
    n_tl = len(tl.get("accuracy", []))
    n_ft = len(ft.get("accuracy", []))
    x_tl = list(range(1, n_tl + 1))
    x_ft = list(range(n_tl + 1, n_tl + n_ft + 1))
    if tl:
        ax.plot(x_tl, tl["accuracy"], "b-", label="Train (TL)")
        ax.plot(x_tl, tl["val_accuracy"], "b--", label="Valid (TL)")
    if ft:
        ax.plot(x_ft, ft["accuracy"], "g-", label="Train (FT)")
        ax.plot(x_ft, ft["val_accuracy"], "g--", label="Valid (FT)")
    if n_tl:
        ax.axvline(x=n_tl + 0.5, color="gray", linestyle=":",
                alpha=0.5, label="Inicio Fine-Tuning")
    ax.set_xlabel("Época")
    ax.set_ylabel("Accuracy")
    ax.set_title("Evolución de la exactitud durante el entrenamiento")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)

    # ----- loss -----
    ax = axes[1]
    if tl:
        ax.plot(x_tl, tl["loss"], "b-", label="Train (TL)")
        ax.plot(x_tl, tl["val_loss"], "b--", label="Valid (TL)")
    if ft:
        ax.plot(x_ft, ft["loss"], "g-", label="Train (FT)")
        ax.plot(x_ft, ft["val_loss"], "g--", label="Valid (FT)")
    if n_tl:
        ax.axvline(x=n_tl + 0.5, color="gray", linestyle=":",
                alpha=0.5, label="Inicio Fine-Tuning")
    ax.set_xlabel("Época")
    ax.set_ylabel("Loss")
    ax.set_title("Evolución de la pérdida durante el entrenamiento")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    p = argparse.ArgumentParser(description="Evalua el modelo entrenado")
    p.add_argument("--arq", default="mobilenet",
                help="Arquitectura: mobilenet (default) o efficient")
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--batch", type=int, default=16)
    args = p.parse_args()

    REPORTE_DIR.mkdir(parents=True, exist_ok=True)
    modelo_path = MODELS_DIR / f"{args.arq}_billetes.keras"
    history_path = MODELS_DIR / f"{args.arq}_billetes_history.json"

    if not modelo_path.exists():
        print(f"ERROR: no existe {modelo_path}")
        print("Corre primero: python scripts/06_entrenar_modelo.py")
        return 1

    out = []
    def log(msg=""):
        print(msg); out.append(msg)

    log("=" * 70)
    log(f"  SCRIPT 07 - EVALUACION DEL MODELO ({args.arq.upper()})")
    log("=" * 70)
    log()

    # ------- cargar modelo y datos -------
    log("Cargando modelo y conjunto de prueba...")
    modelo = tf.keras.models.load_model(modelo_path)
    test_ds = cargar_test_set(args.img_size, args.batch)
    log(f"  Modelo : {modelo_path.name}")
    log(f"  Test   : {sum(1 for _ in test_ds.unbatch())} imagenes")
    log()

    # ------- predicciones -------
    log("Generando predicciones...")
    y_true, y_pred = predecir_todo(modelo, test_ds)
    log(f"  Predicciones: {len(y_pred)}")
    log()

    # ------- metricas globales -------
    test_loss, test_acc = modelo.evaluate(test_ds, verbose=0)
    log("METRICAS GLOBALES")
    log("-" * 40)
    log(f"  Accuracy : {test_acc:.4f}  ({test_acc * 100:.2f}%)")
    log(f"  Loss     : {test_loss:.4f}")
    log()

    # ------- metricas por clase -------
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(CLASES))), zero_division=0,
    )
    log("METRICAS POR CLASE")
    log("-" * 60)
    log(f"  {'CLASE':<14}{'PREC':>8}{'RECALL':>8}{'F1':>8}{'SOPORTE':>10}")
    log("  " + "-" * 50)
    for i, c in enumerate(CLASES):
        log(f"  {c:<14}{precision[i]:>8.3f}{recall[i]:>8.3f}{f1[i]:>8.3f}{support[i]:>10d}")
    log()

    # ------- promedios -------
    p_macro, r_macro, f_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0,
    )
    p_w, r_w, f_w, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0,
    )
    log("PROMEDIOS")
    log("-" * 40)
    log(f"  Macro    -> Prec: {p_macro:.3f}  Recall: {r_macro:.3f}  F1: {f_macro:.3f}")
    log(f"  Weighted -> Prec: {p_w:.3f}  Recall: {r_w:.3f}  F1: {f_w:.3f}")
    log()

    # ------- guardar CSV de metricas -------
    csv_path = REPORTE_DIR / "07_metricas_por_clase.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["clase", "precision", "recall", "f1", "soporte"])
        for i, c in enumerate(CLASES):
            w.writerow([c, f"{precision[i]:.4f}", f"{recall[i]:.4f}",
                        f"{f1[i]:.4f}", int(support[i])])
        w.writerow(["", "", "", "", ""])
        w.writerow(["macro_avg", f"{p_macro:.4f}", f"{r_macro:.4f}", f"{f_macro:.4f}", len(y_true)])
        w.writerow(["weighted_avg", f"{p_w:.4f}", f"{r_w:.4f}", f"{f_w:.4f}", len(y_true)])
        w.writerow(["accuracy", "", "", f"{test_acc:.4f}", len(y_true)])
    log(f"  Metricas CSV: {csv_path.name}")

    # ------- matriz de confusion -------
    cm_path = REPORTE_DIR / "07_matriz_confusion.png"
    cm = graficar_matriz_confusion(y_true, y_pred, CLASES, cm_path)
    log(f"  Matriz confusion: {cm_path.name}")

    # ------- curvas de entrenamiento -------
    if history_path.exists():
        with open(history_path, encoding="utf-8") as f:
            history_data = json.load(f)
        curvas_path = REPORTE_DIR / "07_curvas_entrenamiento.png"
        graficar_curvas(history_data, curvas_path)
        log(f"  Curvas entrenamiento: {curvas_path.name}")
    else:
        log(f"  [!] No se encontro {history_path.name} (omitiendo curvas)")

    # ------- diagnostico de errores mas comunes -------
    log()
    log("ERRORES MAS COMUNES (donde el modelo se confunde)")
    log("-" * 60)
    errores = []
    for i in range(len(CLASES)):
        for j in range(len(CLASES)):
            if i != j and cm[i][j] > 0:
                errores.append((cm[i][j], CLASES[i], CLASES[j]))
    errores.sort(reverse=True)
    if errores:
        for cnt, real, pred in errores[:10]:
            log(f"  {cnt} veces predijo '{pred}' cuando era '{real}'")
    else:
        log("  No hay errores en el conjunto de prueba (perfecto).")
    log()

    # ------- recomendaciones para el informe -------
    log("=" * 70)
    log("INTERPRETACION PARA EL INFORME")
    log("=" * 70)
    if test_acc >= 0.92:
        log(f"  El modelo alcanza {test_acc * 100:.2f}% de exactitud sobre el conjunto")
        log(f"  de prueba, superando el criterio de exito definido en el marco")
        log(f"  practico (>=92%). El sistema satisface las expectativas planteadas")
        log(f"  para la fase de identificacion de denominacion.")
    elif test_acc >= 0.85:
        log(f"  El modelo alcanza {test_acc * 100:.2f}% de exactitud sobre el conjunto")
        log(f"  de prueba, dentro del rango aceptable (85-92%) para un dataset")
        log(f"  de tamano limitado. Las metricas son defendibles para una tesis")
        log(f"  de pregrado.")
    else:
        log(f"  El modelo alcanza {test_acc * 100:.2f}% de exactitud, por debajo")
        log(f"  del objetivo. Considere ampliar el dataset o reentrenar con mas")
        log(f"  epocas / hiperparametros distintos.")

    # guardar reporte texto
    report_path = REPORTE_DIR / "07_evaluacion_reporte.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(out))

    log()
    log(f"Reporte texto: {report_path.name}")
    log()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
