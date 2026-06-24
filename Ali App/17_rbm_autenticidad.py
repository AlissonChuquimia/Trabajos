"""
=============================================================================
SCRIPT 17 - MÁQUINAS DE BOLTZMANN RESTRINGIDAS (RBM) APLICADAS AL PROYECTO
=============================================================================
¿QUÉ ES UNA RBM?
  Una Máquina de Boltzmann Restringida es una red neuronal generativa y
  no supervisada con dos capas:
    - Capa visible (v): las neuronas de entrada (píxeles del billete)
    - Capa oculta (h): las neuronas que aprenden características latentes

  "Restringida" significa que NO hay conexiones dentro de una misma capa
  (a diferencia de una Boltzmann Machine completa). Esto simplifica el
  entrenamiento usando el algoritmo Contrastive Divergence (CD).

  ENERGÍA del sistema (por eso "Boltzmann" — termodinámica estadística):
    E(v,h) = -bᵀv - cᵀh - vᵀWh
    donde:
      v = vector visible (píxeles normalizados)
      h = vector oculto (características aprendidas)
      W = matriz de pesos entre capas
      b = sesgos de la capa visible
      c = sesgos de la capa oculta

  La RBM aprende la distribución de probabilidad P(v) de los datos de
  entrenamiento. Cuando le mostramos un billete AUTÉNTICO, el modelo puede
  reconstruirlo con bajo error. Cuando le mostramos un billete FALSO, el
  error de reconstrucción es alto porque la distribución aprendida no lo
  cubre bien.

¿CÓMO APLICA AL PROYECTO ALI?
  El proyecto ya tiene dos modelos CNN (supervisados):
    - Modelo A: clasifica denominación (bs10...bs200)
    - Modelo B: clasifica auténtico/falso (supervisado con etiquetas)

  La RBM ofrece una perspectiva DIFERENTE y COMPLEMENTARIA:
    - Es NO SUPERVISADA: aprende solo de billetes auténticos (sin etiquetas)
    - Detecta falsificaciones midiendo el ERROR DE RECONSTRUCCIÓN
    - Si la RBM no puede reconstruir una imagen bien → probablemente es falsa

  Ventaja clave: la RBM puede detectar tipos de falsificaciones que NO
  están en el dataset de entrenamiento (billetes falsos nunca vistos antes),
  porque mide "qué tan diferente es esto de lo que aprendí como auténtico".

ESTRUCTURA DEL SCRIPT:
  1. Preprocesamiento: cargar imágenes, aplanar, normalizar a [0,1]
  2. Entrenamiento: RBM entrenada SOLO con billetes auténticos de train
  3. Evaluación: medir error de reconstrucción en test (auténticos y falsos)
  4. Métricas: ROC, AUC, umbral óptimo
  5. Visualización: características aprendidas (receptive fields)
  6. Comparación con Modelo B (CNN supervisado)

USO:
  .venv\\Scripts\\activate
  python scripts/17_rbm_autenticidad.py

REQUISITOS:
  scikit-learn, numpy, pillow, matplotlib
=============================================================================
"""

import sys
import io
import math
import time
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image, ImageOps
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    roc_curve, auc, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# =============================================================================
# CONFIGURACIÓN
# =============================================================================
BASE_DIR    = Path(__file__).parent.parent
DS_AUT      = BASE_DIR / "dataset_autenticidad"
REPORTE_DIR = BASE_DIR / "reportes"
REPORTE_DIR.mkdir(exist_ok=True)

# Tamaño al que reducimos cada imagen antes de aplanar.
# La RBM trabaja con vectores 1D → más pequeño = más rápido de entrenar.
# 64x64 = 4096 píxeles por imagen (en escala de grises).
# Con color RGB serían 12288 dimensiones, demasiado para RBM básica.
IMG_SIZE     = 64       # px (cuadrado, escala de grises)
N_VISIBLE    = IMG_SIZE * IMG_SIZE  # 4096 neuronas visibles

# Hiperparámetros de la RBM
N_HIDDEN     = 256      # neuronas ocultas (características latentes)
N_ITER       = 50       # épocas de entrenamiento Contrastive Divergence
LR           = 0.01     # tasa de aprendizaje
BATCH_SIZE   = 32       # batch para CD
RANDOM_STATE = 42


# =============================================================================
# FUNCIONES DE CARGA Y PREPROCESAMIENTO
# =============================================================================
def cargar_imagen(ruta: Path) -> np.ndarray | None:
    """
    Carga una imagen, la convierte a escala de grises y la redimensiona.
    Devuelve un vector 1D normalizado en [0, 1].
    La normalización es importante para BernoulliRBM que espera valores
    entre 0 y 1 (interpreta las entradas como probabilidades de activación).
    """
    try:
        img = Image.open(ruta).convert("L")   # escala de grises
        img = ImageOps.exif_transpose(img)    # corregir orientación EXIF
        img = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
        arr = np.array(img, dtype=np.float32) / 255.0  # normalizar [0,1]
        return arr.flatten()   # vector de 4096 dimensiones
    except Exception:
        return None


def cargar_split(split: str, verbose: bool = True):
    """
    Carga todas las imágenes de un split (train/valid/test).
    Retorna (X, y) donde y=0 auténtico, y=1 falso.
    """
    X, y = [], []
    for etiqueta, clase in [(0, "autentico"), (1, "falso")]:
        folder = DS_AUT / split / clase
        if not folder.exists():
            continue
        imagenes = sorted(folder.glob("*.jpg"))
        ok = 0
        for ruta in imagenes:
            vec = cargar_imagen(ruta)
            if vec is not None:
                X.append(vec)
                y.append(etiqueta)
                ok += 1
        if verbose:
            print(f"  {split}/{clase:<12}: {ok} imágenes cargadas")
    return np.array(X, dtype=np.float32), np.array(y, dtype=int)


# =============================================================================
# FUNCIÓN DE ENERGÍA LIBRE (Free Energy)
# =============================================================================
def energia_libre(rbm: BernoulliRBM, v: np.ndarray) -> np.ndarray:
    """
    Calcula la energía libre de cada muestra visible v.

    Fórmula:
      F(v) = -bᵀv - Σᵢ log(1 + exp(cᵢ + Wᵢᵀv))

    La energía libre es proporcional a -log P(v):
      - Energía BAJA → imagen muy probable bajo el modelo → auténtico
      - Energía ALTA  → imagen poco probable bajo el modelo → posible falso

    Esta es la métrica principal para detectar anomalías con RBM.
    """
    # Transformar a espacio oculto
    wx_b = np.dot(v, rbm.components_.T) + rbm.intercept_hidden_  # (n, n_hidden)
    # Energía libre: -bv - sum_j log(1 + exp(W_j·v + c_j))
    energia = (
        -np.dot(v, rbm.intercept_visible_)
        - np.sum(np.log1p(np.exp(wx_b)), axis=1)
    )
    return energia


def error_reconstruccion(rbm: BernoulliRBM, v: np.ndarray) -> np.ndarray:
    """
    Error cuadrático medio entre la imagen original y su reconstrucción.
    La RBM reconstruye: v → h = P(h|v) → v' = P(v|h)
    El error mide cuánto difiere v' de v.
    Auténticos → error bajo. Falsos → error alto (idealmente).
    """
    h_prob  = rbm.transform(v)          # P(h|v): activaciones ocultas
    # Reconstrucción: P(v|h) usando los pesos transpuestos
    v_recon = 1.0 / (1.0 + np.exp(
        -(np.dot(h_prob, rbm.components_) + rbm.intercept_visible_)
    ))
    # Error cuadrático medio por imagen
    return np.mean((v - v_recon) ** 2, axis=1)


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("=" * 70)
    print("  SCRIPT 17 - RBM PARA DETECCIÓN DE BILLETES FALSOS (APP ALI)")
    print("=" * 70)
    print()
    print("  ¿Qué hace este script?")
    print("  Entrena una Máquina de Boltzmann Restringida (RBM) usando")
    print("  SOLO billetes auténticos. Luego mide cuánto error comete al")
    print("  intentar reconstruir billetes auténticos vs falsos.")
    print("  Mayor error = mayor probabilidad de ser falso.")
    print()

    # ── 1. CARGAR DATOS ──────────────────────────────────────────────────────
    print("=" * 70)
    print("  PASO 1: Cargando imágenes del dataset de autenticidad")
    print("=" * 70)

    print("\n  Train:")
    X_train, y_train = cargar_split("train")
    print("\n  Valid:")
    X_valid, y_valid = cargar_split("valid")
    print("\n  Test:")
    X_test,  y_test  = cargar_split("test")

    # RBM entrena SOLO con auténticos (aprendizaje no supervisado de auténticos)
    X_train_aut = X_train[y_train == 0]
    print(f"\n  Imágenes para entrenar RBM (solo auténticos): {len(X_train_aut)}")
    print(f"  Dimensión de cada vector: {X_train_aut.shape[1]} ({IMG_SIZE}x{IMG_SIZE} px)")
    print(f"  Rango de valores: [{X_train_aut.min():.3f}, {X_train_aut.max():.3f}]")

    # ── 2. CONSTRUIR Y ENTRENAR LA RBM ───────────────────────────────────────
    print()
    print("=" * 70)
    print("  PASO 2: Entrenando la RBM")
    print("=" * 70)
    print()
    print(f"  Arquitectura:")
    print(f"    Capa visible  : {N_VISIBLE} neuronas  ({IMG_SIZE}x{IMG_SIZE} píxeles)")
    print(f"    Capa oculta   : {N_HIDDEN} neuronas  (características latentes)")
    print(f"    Pesos W       : {N_VISIBLE}×{N_HIDDEN} = {N_VISIBLE*N_HIDDEN:,} parámetros")
    print(f"    Algoritmo     : Contrastive Divergence (CD-1)")
    print(f"    Épocas        : {N_ITER}")
    print(f"    Learning rate : {LR}")
    print(f"    Batch size    : {BATCH_SIZE}")
    print()

    rbm = BernoulliRBM(
        n_components  = N_HIDDEN,
        learning_rate = LR,
        n_iter        = N_ITER,
        batch_size    = BATCH_SIZE,
        random_state  = RANDOM_STATE,
        verbose       = True,   # muestra pseudo-likelihood por época
    )

    t0 = time.time()
    rbm.fit(X_train_aut)
    t_total = time.time() - t0
    print(f"\n  Entrenamiento completado en {t_total:.1f} segundos")

    # ── 3. CALCULAR MÉTRICAS DE ANOMALÍA ────────────────────────────────────
    print()
    print("=" * 70)
    print("  PASO 3: Midiendo error de reconstrucción en test")
    print("=" * 70)
    print()
    print("  Concepto: si la RBM aprendió bien los billetes auténticos,")
    print("  reconstruirá los auténticos con bajo error y los falsos con")
    print("  alto error (porque nunca vio patrones falsos en entrenamiento).")
    print()

    # Error de reconstrucción en test
    err_test = error_reconstruccion(rbm, X_test)
    energia_test = energia_libre(rbm, X_test)

    err_aut  = err_test[y_test == 0]
    err_fals = err_test[y_test == 1]
    eng_aut  = energia_test[y_test == 0]
    eng_fals = energia_test[y_test == 1]

    print(f"  {'Métrica':<30} {'Auténticos':>12} {'Falsos':>12}")
    print(f"  {'-'*56}")
    print(f"  {'Error reconstrucción (media)':<30} "
          f"{err_aut.mean():>12.5f} {err_fals.mean():>12.5f}")
    print(f"  {'Error reconstrucción (min)':<30} "
          f"{err_aut.min():>12.5f} {err_fals.min():>12.5f}")
    print(f"  {'Error reconstrucción (max)':<30} "
          f"{err_aut.max():>12.5f} {err_fals.max():>12.5f}")
    print(f"  {'Energía libre (media)':<30} "
          f"{eng_aut.mean():>12.2f} {eng_fals.mean():>12.2f}")
    print()

    separacion = err_fals.mean() - err_aut.mean()
    print(f"  Separación (err_falso - err_auténtico): {separacion:.5f}")
    if separacion > 0:
        print("  ✓ La RBM asigna MAYOR error a los falsos → puede detectarlos")
    else:
        print("  ✗ La RBM no logra separar bien → el dataset necesita más variedad")

    # ── 4. ROC Y AUC ─────────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("  PASO 4: Curva ROC — ¿qué tan bien separa auténticos de falsos?")
    print("=" * 70)

    # Score = error de reconstrucción: alto error → probable falso (y=1)
    fpr, tpr, umbrales = roc_curve(y_test, err_test)
    roc_auc = auc(fpr, tpr)

    # Umbral óptimo: maximiza TPR - FPR (índice de Youden)
    j_scores = tpr - fpr
    idx_opt  = np.argmax(j_scores)
    umbral_opt    = umbrales[idx_opt]
    tpr_opt       = tpr[idx_opt]
    fpr_opt       = fpr[idx_opt]

    print(f"\n  AUC-ROC : {roc_auc:.4f}  (1.0 = perfecto, 0.5 = aleatorio)")
    print(f"  Umbral óptimo de error: {umbral_opt:.5f}")
    print(f"    → Con este umbral: TPR={tpr_opt:.3f} (detecta {tpr_opt*100:.1f}% de falsos)")
    print(f"    →                 FPR={fpr_opt:.3f} (falsa alarma en {fpr_opt*100:.1f}% de auténticos)")

    # Clasificación con umbral óptimo
    y_pred_rbm = (err_test >= umbral_opt).astype(int)
    print()
    print("  Reporte de clasificación con umbral óptimo:")
    print(classification_report(y_test, y_pred_rbm,
                                target_names=["autentico","falso"],
                                digits=4))

    # ── 5. VISUALIZACIONES ───────────────────────────────────────────────────
    print("=" * 70)
    print("  PASO 5: Generando visualizaciones")
    print("=" * 70)

    fig = plt.figure(figsize=(18, 14))
    fig.suptitle(
        "RBM para Detección de Billetes Falsos — APP ALI\n"
        "Máquina de Boltzmann Restringida (aprendizaje no supervisado)",
        fontsize=14, fontweight="bold", y=0.98
    )
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    # ── 5a. Características aprendidas (receptive fields) ──────────────────
    ax_rf = fig.add_subplot(gs[0, :])
    ax_rf.set_title(
        "Características aprendidas por la RBM (primeras 32 neuronas ocultas)\n"
        "Cada cuadro = qué patrón visual activa esa neurona oculta",
        fontsize=11
    )
    n_show = 32
    w = rbm.components_[:n_show].reshape(n_show, IMG_SIZE, IMG_SIZE)
    w_norm = (w - w.min()) / (w.max() - w.min() + 1e-8)
    grid_img = np.zeros((IMG_SIZE, n_show * (IMG_SIZE + 2)))
    for i in range(n_show):
        grid_img[:, i*(IMG_SIZE+2):(i*(IMG_SIZE+2)+IMG_SIZE)] = w_norm[i]
    ax_rf.imshow(grid_img, cmap="viridis", aspect="auto")
    ax_rf.axis("off")

    # ── 5b. Histograma de errores ──────────────────────────────────────────
    ax_hist = fig.add_subplot(gs[1, 0])
    ax_hist.hist(err_aut,  bins=30, alpha=0.7, color="#3daf7a",
                 label=f"Auténticos (n={len(err_aut)})", density=True)
    ax_hist.hist(err_fals, bins=30, alpha=0.7, color="#e05c5c",
                 label=f"Falsos (n={len(err_fals)})",    density=True)
    ax_hist.axvline(umbral_opt, color="navy", linestyle="--", lw=2,
                    label=f"Umbral óptimo\n({umbral_opt:.4f})")
    ax_hist.set_title("Distribución del error\nde reconstrucción (test)", fontsize=10)
    ax_hist.set_xlabel("Error cuadrático medio")
    ax_hist.set_ylabel("Densidad")
    ax_hist.legend(fontsize=8)
    ax_hist.grid(alpha=0.3)

    # ── 5c. Curva ROC ──────────────────────────────────────────────────────
    ax_roc = fig.add_subplot(gs[1, 1])
    ax_roc.plot(fpr, tpr, color="#4f79c4", lw=2,
                label=f"RBM (AUC = {roc_auc:.4f})")
    ax_roc.plot([0,1], [0,1], "k--", lw=1, label="Aleatorio (AUC=0.50)")
    ax_roc.scatter([fpr_opt], [tpr_opt], color="red", zorder=5,
                   label=f"Umbral óptimo\nTPR={tpr_opt:.3f} FPR={fpr_opt:.3f}")
    ax_roc.set_title("Curva ROC\n(error de reconstrucción como score)", fontsize=10)
    ax_roc.set_xlabel("Tasa de falsos positivos (FPR)")
    ax_roc.set_ylabel("Tasa de verdaderos positivos (TPR)")
    ax_roc.legend(fontsize=8)
    ax_roc.grid(alpha=0.3)

    # ── 5d. Matriz de confusión RBM ────────────────────────────────────────
    ax_cm = fig.add_subplot(gs[1, 2])
    cm = confusion_matrix(y_test, y_pred_rbm)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Auténtico","Falso"])
    disp.plot(ax=ax_cm, colorbar=False, cmap="Blues")
    ax_cm.set_title("Matriz de confusión\nRBM (umbral óptimo)", fontsize=10)

    # ── 5e. Ejemplos: reconstrucción auténtico vs falso ────────────────────
    ax_ej = fig.add_subplot(gs[2, :])
    ax_ej.axis("off")
    ax_ej.set_title(
        "Reconstrucción de imágenes: auténticos (verde) vs falsos (rojo)\n"
        "Original (arriba) → Reconstrucción RBM (abajo) — mayor diferencia visual = mayor error",
        fontsize=10
    )

    n_ej = 6  # 3 auténticos + 3 falsos
    idx_aut  = np.where(y_test == 0)[0][:3]
    idx_fals = np.where(y_test == 1)[0][:3]
    indices_ej = list(idx_aut) + list(idx_fals)
    colores_ej = ["#3daf7a"]*3 + ["#e05c5c"]*3

    h_probs = rbm.transform(X_test[indices_ej])
    v_recon = 1.0 / (1.0 + np.exp(
        -(np.dot(h_probs, rbm.components_) + rbm.intercept_visible_)
    ))

    for col, idx_ej in enumerate(range(n_ej)):
        # Original
        ax_o = fig.add_axes([0.04 + col*0.155, 0.07, 0.13, 0.11])
        ax_o.imshow(X_test[indices_ej[idx_ej]].reshape(IMG_SIZE, IMG_SIZE),
                    cmap="gray", vmin=0, vmax=1)
        err_ej = err_test[indices_ej[idx_ej]]
        ax_o.set_title(f"{'AUT' if idx_ej<3 else 'FALSO'}\nerr={err_ej:.4f}",
                       fontsize=8, color=colores_ej[idx_ej], fontweight="bold")
        ax_o.axis("off")
        for spine in ax_o.spines.values():
            spine.set_edgecolor(colores_ej[idx_ej]); spine.set_linewidth(3)

        # Reconstrucción
        ax_r = fig.add_axes([0.04 + col*0.155, 0.01, 0.13, 0.055])
        ax_r.imshow(v_recon[idx_ej].reshape(IMG_SIZE, IMG_SIZE),
                    cmap="gray", vmin=0, vmax=1)
        ax_r.set_title("Reconstr.", fontsize=7)
        ax_r.axis("off")

    ruta_fig = REPORTE_DIR / "17_rbm_resultados.png"
    plt.savefig(ruta_fig, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"  Figura guardada: {ruta_fig}")

    # ── 6. COMPARACIÓN CON MODELO B CNN ─────────────────────────────────────
    print()
    print("=" * 70)
    print("  PASO 6: Comparación — RBM vs Modelo B (CNN supervisado)")
    print("=" * 70)
    print()
    print("  ┌─────────────────────────────┬──────────────────┬──────────────────┐")
    print("  │ Característica              │ RBM              │ Modelo B (CNN)   │")
    print("  ├─────────────────────────────┼──────────────────┼──────────────────┤")
    print("  │ Tipo de aprendizaje         │ No supervisado   │ Supervisado      │")
    print("  │ Necesita etiquetas          │ NO               │ SÍ               │")
    print("  │ Entrena con falsos          │ NO (solo aut.)   │ SÍ               │")
    print("  │ Detecta falsos NO vistos    │ SÍ (anomalía)    │ Parcialmente     │")
    print("  │ Resolución de entrada       │ 64×64 gris       │ 224×224 color    │")
    print("  │ Parámetros                  │ {:>10,}   │ ~2,200,000       │".format(N_VISIBLE*N_HIDDEN))
    print(f"  │ AUC-ROC en test             │ {roc_auc:.4f}           │ ~0.97 (Modelo B) │")
    print("  │ Velocidad de inferencia     │ Muy rápida       │ Moderada (TFLite)│")
    print("  │ Interpretabilidad           │ Alta (campos rec)│ Baja (caja negra)│")
    print("  └─────────────────────────────┴──────────────────┴──────────────────┘")
    print()
    print("  CONCLUSIÓN:")
    if roc_auc >= 0.80:
        print(f"  La RBM alcanza AUC={roc_auc:.4f}, lo cual es NOTABLE para un modelo")
        print("  no supervisado que NUNCA vio billetes falsos durante el entrenamiento.")
        print("  Puede funcionar como primera capa de detección de anomalías,")
        print("  complementando al Modelo B CNN para tipos de falsificación nuevos.")
    elif roc_auc >= 0.65:
        print(f"  La RBM alcanza AUC={roc_auc:.4f}. Hay separación entre auténticos y")
        print("  falsos, pero limitada. El Modelo B CNN supervisado es superior para")
        print("  el uso en producción. La RBM es útil para casos de falsificaciones")
        print("  NO vistas durante el entrenamiento del Modelo B.")
    else:
        print(f"  La RBM alcanza AUC={roc_auc:.4f}. La separación es limitada con el")
        print("  dataset actual. Esto ocurre porque las falsificaciones (alasitas)")
        print("  comparten patrones visuales básicos con los billetes auténticos.")
        print("  El Modelo B CNN supervisado es claramente superior para este dataset.")

    print()
    print("  USO PRÁCTICO en APP ALI:")
    print("  La RBM podría usarse como FILTRO PREVIO al Modelo B:")
    print("    1. RBM evalúa error de reconstrucción (muy rápido, sin GPU)")
    print("    2. Si error > umbral_opt → FALSO directo (sin gastar Modelo B)")
    print("    3. Si error <= umbral_opt → pasa al Modelo B para confirmación")
    print("  Esto reduce la carga computacional en casos obvios (alasitas).")

    # Guardar resumen JSON
    resumen = {
        "modelo": "BernoulliRBM",
        "n_visible": N_VISIBLE,
        "n_hidden": N_HIDDEN,
        "n_iter": N_ITER,
        "img_size": IMG_SIZE,
        "n_train_autenticos": int(len(X_train_aut)),
        "n_test_total": int(len(y_test)),
        "auc_roc": float(roc_auc),
        "umbral_optimo": float(umbral_opt),
        "tpr_opt": float(tpr_opt),
        "fpr_opt": float(fpr_opt),
        "err_media_autenticos": float(err_aut.mean()),
        "err_media_falsos": float(err_fals.mean()),
        "separacion_err": float(separacion),
        "figura": str(REPORTE_DIR / "17_rbm_resultados.png"),
    }
    json_path = REPORTE_DIR / "17_rbm_resumen.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(resumen, f, indent=2)

    print()
    print("=" * 70)
    print(f"  Resumen guardado : {json_path}")
    print(f"  Figura guardada  : {ruta_fig}")
    print("=" * 70)


if __name__ == "__main__":
    main()
