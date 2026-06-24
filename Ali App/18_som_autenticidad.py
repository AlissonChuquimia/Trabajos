"""
=============================================================================
SCRIPT 18 - MAPAS AUTOORGANIZADOS (SOM) APLICADOS AL PROYECTO ALI
=============================================================================
¿QUÉ ES UN SOM?
  Un Mapa Autoorganizado (Self-Organizing Map) es una red neuronal NO
  supervisada que aprende a representar datos de alta dimensión en una
  cuadrícula 2D preservando la topología del espacio original.

  Inventado por Teuvo Kohonen (1982), también se llama "red de Kohonen".

  ESTRUCTURA:
    - Cuadrícula de neuronas (ej: 10×10 = 100 neuronas)
    - Cada neurona tiene un vector de pesos W de la misma dimensión
      que los datos de entrada (en nuestro caso: 4096 = 64×64 px)
    - Las neuronas vecinas en el mapa aprenden patrones similares

  ENTRENAMIENTO (Aprendizaje competitivo):
    Para cada imagen de entrenamiento:
      1. Encontrar la neurona más cercana → BMU (Best Matching Unit)
         BMU = argmin ||x - W_i||  (distancia euclidiana mínima)
      2. Actualizar el BMU y sus vecinos:
         W_i(t+1) = W_i(t) + η(t) · h(BMU,i,t) · [x - W_i(t)]
         donde:
           η(t)        = tasa de aprendizaje (decrece con el tiempo)
           h(BMU,i,t)  = función de vecindad (gaussiana, decrece con distancia)
      3. Con el tiempo el radio de vecindad decrece → neuronas se especializan

  DETECCIÓN DE ANOMALÍAS con SOM:
    Una vez entrenado con billetes AUTÉNTICOS:
    - Error de cuantización (QE) = ||x - W_BMU|| (distancia al BMU)
    - Auténticos → QE bajo  (imagen cae cerca de una neurona conocida)
    - Falsos     → QE alto  (imagen cae lejos de todo lo aprendido)

  DIFERENCIA CON RBM:
    - RBM: modelo generativo probabilístico (energía de Boltzmann)
    - SOM: mapa topográfico 2D (preserva estructura espacial de los datos)
    - SOM produce una visualización 2D interpretable del espacio de billetes
    - SOM muestra QUÉ denominaciones son visualmente similares entre sí

¿CÓMO APLICA AL PROYECTO ALI?
  1. DETECCIÓN DE ANOMALÍAS:
     Igual que la RBM, pero usando distancia euclidiana al BMU como score.
     Útil para detectar falsificaciones nunca vistas.

  2. VISUALIZACIÓN DE DENOMINACIONES:
     Si entrenamos el SOM con billetes de todas las denominaciones,
     el mapa muestra qué billetes son visualmente parecidos.
     Billetes del mismo color (bs10 azul, bs50 morado) deberían
     caer en zonas adyacentes del mapa.

  3. ANÁLISIS DE CONFUSIÓN:
     Las denominaciones que se confunden en el Modelo A (bs20/bs50)
     deberían quedar cerca en el mapa SOM → explica visualmente
     por qué el CNN las confunde.

ESTRUCTURA DEL SCRIPT:
  1. Preprocesamiento: imágenes 64×64 px escala de grises, normalizadas
  2. SOM para AUTENTICIDAD: entrenado solo con auténticos
     → detecta falsos por error de cuantización
  3. SOM para DENOMINACIÓN: entrenado con todas las clases
     → visualiza topología de las denominaciones
  4. Métricas: ROC, AUC, umbral óptimo (igual que RBM para comparar)
  5. Visualizaciones: U-Matrix, mapa de hits, mapa de denominaciones

USO:
  .venv\\Scripts\\activate
  python scripts/18_som_autenticidad.py

REQUISITOS:
  minisom, numpy, pillow, matplotlib, scikit-learn
=============================================================================
"""

import sys
import io
import json
import time
import math
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from PIL import Image, ImageOps
from minisom import MiniSom
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
DS_CLAS     = BASE_DIR / "dataset_clasificacion"
REPORTE_DIR = BASE_DIR / "reportes"
REPORTE_DIR.mkdir(exist_ok=True)

IMG_SIZE    = 64          # px cuadrado, escala de grises
N_VISIBLE   = IMG_SIZE * IMG_SIZE  # 4096 dimensiones

# Hiperparámetros SOM para autenticidad
SOM_X       = 12          # columnas de la cuadrícula
SOM_Y       = 12          # filas de la cuadrícula
N_ITER_AUT  = 5000        # iteraciones de entrenamiento
LR_AUT      = 0.5         # tasa de aprendizaje inicial
SIGMA_AUT   = 3.0         # radio de vecindad inicial

# Hiperparámetros SOM para denominación (más pequeño, dataset más grande)
SOM_X_DEN   = 10
SOM_Y_DEN   = 10
N_ITER_DEN  = 3000
LR_DEN      = 0.5
SIGMA_DEN   = 2.5

RANDOM_STATE = 42
CLASES_DEN  = ["bs10", "bs20", "bs50", "bs100", "bs200"]

# Colores reales de cada denominación boliviana
COLORES_DEN = {
    "bs10":  "#3B7FC4",   # azul
    "bs20":  "#F5A623",   # naranja
    "bs50":  "#8B5CF6",   # morado
    "bs100": "#E05C5C",   # rojo
    "bs200": "#92593A",   # marrón
}


# =============================================================================
# PREPROCESAMIENTO
# =============================================================================
def cargar_imagen(ruta: Path) -> np.ndarray | None:
    """
    Carga imagen, convierte a gris 64×64, normaliza a [0,1].
    BernoulliRBM y SOM esperan valores normalizados.
    """
    try:
        img = Image.open(ruta).convert("L")
        img = ImageOps.exif_transpose(img)
        img = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
        arr = np.array(img, dtype=np.float32) / 255.0
        return arr.flatten()
    except Exception:
        return None


def cargar_dataset_autenticidad(split: str, verbose=True):
    X, y = [], []
    for etiqueta, clase in [(0, "autentico"), (1, "falso")]:
        folder = DS_AUT / split / clase
        if not folder.exists():
            continue
        ok = 0
        for ruta in sorted(folder.glob("*.jpg")):
            vec = cargar_imagen(ruta)
            if vec is not None:
                X.append(vec); y.append(etiqueta); ok += 1
        if verbose:
            print(f"  {split}/{clase:<12}: {ok} imágenes")
    return np.array(X, dtype=np.float32), np.array(y, dtype=int)


def cargar_dataset_denominacion(split: str, verbose=True):
    X, y, nombres = [], [], []
    for idx, clase in enumerate(CLASES_DEN):
        folder = DS_CLAS / split / clase
        if not folder.exists():
            continue
        ok = 0
        for ruta in sorted(folder.glob("*.jpg")):
            vec = cargar_imagen(ruta)
            if vec is not None:
                X.append(vec); y.append(idx); nombres.append(clase); ok += 1
        if verbose:
            print(f"  {split}/{clase:<8}: {ok} imágenes")
    return np.array(X, dtype=np.float32), np.array(y, dtype=int), nombres


# =============================================================================
# MÉTRICAS SOM
# =============================================================================
def error_cuantizacion(som: MiniSom, X: np.ndarray) -> np.ndarray:
    """
    Error de cuantización por muestra = distancia euclidiana al BMU.
    Es la métrica principal para detección de anomalías con SOM:
      - Auténticos entrenados: QE bajo (imagen conocida por el mapa)
      - Falsos no vistos: QE alto (imagen extraña para el mapa)
    """
    errores = np.array([
        np.linalg.norm(x - som.get_weights()[som.winner(x)])
        for x in X
    ])
    return errores


# =============================================================================
# VISUALIZACIONES SOM
# =============================================================================
def plot_umatrix(som: MiniSom, ax, title="U-Matrix"):
    """
    U-Matrix: distancia entre neuronas vecinas.
    Zonas oscuras = fronteras entre grupos distintos.
    Zonas claras  = neuronas similares (mismo tipo de billete).
    """
    weights = som.get_weights()
    sx, sy, _ = weights.shape
    umat = np.zeros((sx, sy))
    for i in range(sx):
        for j in range(sy):
            vecinos = []
            for di, dj in [(-1,0),(1,0),(0,-1),(0,1)]:
                ni, nj = i+di, j+dj
                if 0 <= ni < sx and 0 <= nj < sy:
                    vecinos.append(np.linalg.norm(weights[i,j] - weights[ni,nj]))
            umat[i,j] = np.mean(vecinos) if vecinos else 0
    im = ax.imshow(umat.T, cmap="bone_r", interpolation="nearest")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Neurona x"); ax.set_ylabel("Neurona y")
    plt.colorbar(im, ax=ax, shrink=0.8, label="Distancia media a vecinos")
    return umat


def plot_hitmap(som: MiniSom, X: np.ndarray, ax, title="Mapa de hits"):
    """
    Mapa de hits: cuántas imágenes de entrenamiento activan cada neurona.
    Neuronas muy activadas = zonas comunes del espacio de billetes.
    Neuronas no activadas  = zonas vacías del mapa.
    """
    sx, sy = som.get_weights().shape[:2]
    hits = np.zeros((sx, sy))
    for x in X:
        bmu = som.winner(x)
        hits[bmu] += 1
    im = ax.imshow(hits.T, cmap="YlOrRd", interpolation="nearest")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Neurona x"); ax.set_ylabel("Neurona y")
    plt.colorbar(im, ax=ax, shrink=0.8, label="Cantidad de imágenes activadoras")
    return hits


def plot_mapa_denominaciones(som: MiniSom, X: np.ndarray,
                              y: np.ndarray, ax, title="Mapa por denominación"):
    """
    Cada punto = una imagen del dataset mapeada a su BMU.
    El color indica la denominación real.
    Si el SOM funciona bien, billetes del mismo color quedan juntos.
    Billetes visualmente parecidos quedan en zonas adyacentes del mapa.
    """
    ax.set_facecolor("#F0F0F0")
    sx, sy = som.get_weights().shape[:2]

    # Dibujar cuadrícula del SOM
    for i in range(sx+1):
        ax.axvline(i-0.5, color="white", lw=0.5)
    for j in range(sy+1):
        ax.axhline(j-0.5, color="white", lw=0.5)

    # Agregar pequeño jitter para ver puntos superpuestos
    rng = np.random.RandomState(42)
    for xi, yi in zip(X, y):
        bmu = som.winner(xi)
        clase = CLASES_DEN[yi]
        jx = bmu[0] + rng.uniform(-0.35, 0.35)
        jy = bmu[1] + rng.uniform(-0.35, 0.35)
        ax.plot(jx, jy, "o",
                color=COLORES_DEN[clase], markersize=4, alpha=0.6)

    parches = [mpatches.Patch(color=COLORES_DEN[c], label=c) for c in CLASES_DEN]
    ax.legend(handles=parches, fontsize=7, loc="upper right",
              title="Denominación", title_fontsize=8)
    ax.set_xlim(-0.5, sx-0.5); ax.set_ylim(-0.5, sy-0.5)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Neurona x"); ax.set_ylabel("Neurona y")


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("=" * 70)
    print("  SCRIPT 18 - SOM (MAPA AUTOORGANIZADO) — APP ALI")
    print("=" * 70)
    print()
    print("  ¿Qué hace este script?")
    print("  Entrena dos SOMs de Kohonen sobre el dataset de billetes bolivianos:")
    print("  1. SOM de AUTENTICIDAD: entrenado solo con auténticos")
    print("     → detecta falsos por error de cuantización (QE)")
    print("  2. SOM de DENOMINACIÓN: entrenado con todas las clases")
    print("     → visualiza qué denominaciones son visualmente similares")
    print()

    # ─────────────────────────────────────────────────────────────────────────
    # PARTE 1 — SOM DE AUTENTICIDAD
    # ─────────────────────────────────────────────────────────────────────────
    print("=" * 70)
    print("  PARTE 1: SOM DE AUTENTICIDAD")
    print("  (entrenado solo con auténticos, detecta falsos como anomalías)")
    print("=" * 70)

    print("\n  Cargando datos...")
    print("  Train:")
    X_tr, y_tr = cargar_dataset_autenticidad("train")
    print("  Valid:")
    X_va, y_va = cargar_dataset_autenticidad("valid")
    print("  Test:")
    X_te, y_te = cargar_dataset_autenticidad("test")

    X_tr_aut = X_tr[y_tr == 0]   # solo auténticos para entrenar
    print(f"\n  Imágenes de entrenamiento (solo auténticos): {len(X_tr_aut)}")
    print(f"  Dimensión del vector de entrada: {X_tr_aut.shape[1]} ({IMG_SIZE}×{IMG_SIZE})")

    print(f"\n  Arquitectura del SOM:")
    print(f"    Cuadrícula : {SOM_X}×{SOM_Y} = {SOM_X*SOM_Y} neuronas")
    print(f"    Pesos      : {SOM_X}×{SOM_Y}×{N_VISIBLE} = {SOM_X*SOM_Y*N_VISIBLE:,} valores")
    print(f"    Iteraciones: {N_ITER_AUT}")
    print(f"    LR inicial : {LR_AUT}  (decrece linealmente)")
    print(f"    Sigma ini. : {SIGMA_AUT}  (radio de vecindad, decrece)")
    print(f"    Algoritmo  : Aprendizaje competitivo de Kohonen")

    print(f"\n  Entrenando SOM de autenticidad...")
    som_aut = MiniSom(
        x=SOM_X, y=SOM_Y,
        input_len=N_VISIBLE,
        sigma=SIGMA_AUT,
        learning_rate=LR_AUT,
        random_seed=RANDOM_STATE,
    )
    som_aut.random_weights_init(X_tr_aut)

    t0 = time.time()
    som_aut.train(X_tr_aut, num_iteration=N_ITER_AUT, verbose=False)
    print(f"  Entrenamiento completado en {time.time()-t0:.1f} s")

    # Error de cuantización en test
    print("\n  Calculando error de cuantización en test...")
    qe_test = error_cuantizacion(som_aut, X_te)
    qe_aut  = qe_test[y_te == 0]
    qe_fals = qe_test[y_te == 1]

    print(f"\n  {'Métrica':<35} {'Auténticos':>12} {'Falsos':>12}")
    print(f"  {'-'*62}")
    print(f"  {'Error cuantización (QE) medio':<35} "
          f"{qe_aut.mean():>12.5f} {qe_fals.mean():>12.5f}")
    print(f"  {'QE mínimo':<35} "
          f"{qe_aut.min():>12.5f} {qe_fals.min():>12.5f}")
    print(f"  {'QE máximo':<35} "
          f"{qe_aut.max():>12.5f} {qe_fals.max():>12.5f}")

    sep = qe_fals.mean() - qe_aut.mean()
    print(f"\n  Separación (QE_falso - QE_auténtico): {sep:.5f}")
    if sep > 0:
        print("  ✓ SOM asigna MAYOR error de cuantización a los falsos")
    else:
        print("  ✗ SOM no logra separar bien en este dataset")

    # ROC y AUC
    fpr, tpr, umbs = roc_curve(y_te, qe_test)
    roc_auc = auc(fpr, tpr)
    j       = tpr - fpr
    idx_opt = np.argmax(j)
    umbral_opt = umbs[idx_opt]
    tpr_opt    = tpr[idx_opt]
    fpr_opt    = fpr[idx_opt]

    print(f"\n  AUC-ROC : {roc_auc:.4f}")
    print(f"  Umbral óptimo QE: {umbral_opt:.5f}")
    print(f"    TPR = {tpr_opt:.3f}  (detecta {tpr_opt*100:.1f}% de falsos)")
    print(f"    FPR = {fpr_opt:.3f}  (falsa alarma en {fpr_opt*100:.1f}% de auténticos)")

    y_pred_som = (qe_test >= umbral_opt).astype(int)
    print()
    print("  Reporte de clasificación con umbral óptimo:")
    print(classification_report(y_te, y_pred_som,
                                target_names=["autentico","falso"], digits=4))

    # ─────────────────────────────────────────────────────────────────────────
    # PARTE 2 — SOM DE DENOMINACIÓN
    # ─────────────────────────────────────────────────────────────────────────
    print("=" * 70)
    print("  PARTE 2: SOM DE DENOMINACIÓN")
    print("  (visualiza la topología visual de los billetes bolivianos)")
    print("=" * 70)

    print("\n  Cargando dataset de denominación (train)...")
    X_den, y_den, _ = cargar_dataset_denominacion("train", verbose=True)

    print(f"\n  Entrenando SOM de denominación ({SOM_X_DEN}×{SOM_Y_DEN})...")
    som_den = MiniSom(
        x=SOM_X_DEN, y=SOM_Y_DEN,
        input_len=N_VISIBLE,
        sigma=SIGMA_DEN,
        learning_rate=LR_DEN,
        random_seed=RANDOM_STATE,
    )
    som_den.random_weights_init(X_den)
    t0 = time.time()
    som_den.train(X_den, num_iteration=N_ITER_DEN, verbose=False)
    print(f"  Completado en {time.time()-t0:.1f} s")

    # QE promedio por denominación en test
    print("\n  Error de cuantización por denominación (test):")
    X_den_te, y_den_te, _ = cargar_dataset_denominacion("test", verbose=False)
    qe_den_te = error_cuantizacion(som_den, X_den_te)
    print(f"  {'Clase':<8} {'QE medio':>10} {'QE min':>10} {'QE max':>10}")
    print(f"  {'-'*42}")
    for idx, clase in enumerate(CLASES_DEN):
        mask = y_den_te == idx
        if mask.sum() == 0:
            continue
        q = qe_den_te[mask]
        print(f"  {clase:<8} {q.mean():>10.5f} {q.min():>10.5f} {q.max():>10.5f}")

    # ─────────────────────────────────────────────────────────────────────────
    # VISUALIZACIONES
    # ─────────────────────────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("  Generando figura completa de resultados SOM...")
    print("=" * 70)

    fig = plt.figure(figsize=(20, 16))
    fig.suptitle(
        "SOM (Mapa Autoorganizado de Kohonen) — APP ALI\n"
        "Detección de falsificaciones y topología de denominaciones bolivianas",
        fontsize=14, fontweight="bold", y=0.99
    )
    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.5, wspace=0.4)

    # ── Fila 1: SOM autenticidad ──────────────────────────────────────────
    ax_um = fig.add_subplot(gs[0, 0])
    plot_umatrix(som_aut, ax_um,
                 f"U-Matrix — SOM Autenticidad\n({SOM_X}×{SOM_Y} neuronas)")

    ax_hm = fig.add_subplot(gs[0, 1])
    plot_hitmap(som_aut, X_tr_aut, ax_hm,
                "Mapa de Hits\n(auténticos de train)")

    ax_hist = fig.add_subplot(gs[0, 2])
    ax_hist.hist(qe_aut,  bins=30, alpha=0.7, color="#3daf7a",
                 label=f"Auténticos (n={len(qe_aut)})", density=True)
    ax_hist.hist(qe_fals, bins=30, alpha=0.7, color="#e05c5c",
                 label=f"Falsos (n={len(qe_fals)})",    density=True)
    ax_hist.axvline(umbral_opt, color="navy", lw=2, linestyle="--",
                    label=f"Umbral óptimo\n({umbral_opt:.4f})")
    ax_hist.set_title("Error de cuantización (QE)\nAuténticos vs Falsos — test", fontsize=10)
    ax_hist.set_xlabel("QE"); ax_hist.set_ylabel("Densidad")
    ax_hist.legend(fontsize=8); ax_hist.grid(alpha=0.3)

    ax_roc = fig.add_subplot(gs[0, 3])
    ax_roc.plot(fpr, tpr, color="#4f79c4", lw=2,
                label=f"SOM  (AUC={roc_auc:.4f})")
    ax_roc.plot([0,1],[0,1], "k--", lw=1, label="Aleatorio (0.50)")
    ax_roc.scatter([fpr_opt],[tpr_opt], color="red", zorder=5,
                   label=f"Umbral óptimo\nTPR={tpr_opt:.3f}")
    ax_roc.set_title("Curva ROC\nSOM Autenticidad", fontsize=10)
    ax_roc.set_xlabel("FPR"); ax_roc.set_ylabel("TPR")
    ax_roc.legend(fontsize=8); ax_roc.grid(alpha=0.3)

    # ── Fila 2: SOM denominación ──────────────────────────────────────────
    ax_um2 = fig.add_subplot(gs[1, 0])
    plot_umatrix(som_den, ax_um2,
                 f"U-Matrix — SOM Denominación\n({SOM_X_DEN}×{SOM_Y_DEN} neuronas)")

    ax_den = fig.add_subplot(gs[1, 1:3])
    # Usar solo subconjunto del train para no saturar el gráfico
    max_por_clase = 100
    X_vis, y_vis = [], []
    for idx in range(len(CLASES_DEN)):
        mask = np.where(y_den == idx)[0][:max_por_clase]
        X_vis.extend(X_den[mask])
        y_vis.extend(y_den[mask])
    plot_mapa_denominaciones(som_den, np.array(X_vis), np.array(y_vis),
                              ax_den,
                              "Topología de denominaciones en el SOM\n"
                              "(billetes visualmente similares quedan juntos)")

    ax_cm = fig.add_subplot(gs[1, 3])
    cm_mat = confusion_matrix(y_te, y_pred_som)
    ConfusionMatrixDisplay(cm_mat, display_labels=["Auténtico","Falso"]).plot(
        ax=ax_cm, colorbar=False, cmap="Blues"
    )
    ax_cm.set_title("Matriz de confusión\nSOM (umbral óptimo)", fontsize=10)

    # ── Fila 3: Pesos aprendidos por el SOM ──────────────────────────────
    ax_w = fig.add_subplot(gs[2, :])
    ax_w.set_title(
        "Pesos aprendidos por las neuronas del SOM de autenticidad "
        f"(primeras 32 de {SOM_X*SOM_Y} neuronas)\n"
        "Cada cuadro = patrón visual que activa esa neurona",
        fontsize=10
    )
    weights = som_aut.get_weights().reshape(-1, N_VISIBLE)
    n_show  = min(32, len(weights))
    w_norm  = (weights[:n_show] - weights[:n_show].min()) / \
              (weights[:n_show].max() - weights[:n_show].min() + 1e-8)
    grid    = np.zeros((IMG_SIZE, n_show*(IMG_SIZE+2)))
    for i in range(n_show):
        grid[:, i*(IMG_SIZE+2):i*(IMG_SIZE+2)+IMG_SIZE] = \
            w_norm[i].reshape(IMG_SIZE, IMG_SIZE)
    ax_w.imshow(grid, cmap="viridis", aspect="auto")
    ax_w.axis("off")

    ruta_fig = REPORTE_DIR / "18_som_resultados.png"
    plt.savefig(ruta_fig, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Figura guardada: {ruta_fig}")

    # ─────────────────────────────────────────────────────────────────────────
    # COMPARACIÓN FINAL
    # ─────────────────────────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("  COMPARACIÓN FINAL: RBM vs SOM vs Modelo B CNN")
    print("=" * 70)
    print()
    print("  ┌────────────────────────────┬──────────────┬──────────────┬──────────────┐")
    print("  │ Característica             │ SOM          │ RBM          │ Modelo B CNN │")
    print("  ├────────────────────────────┼──────────────┼──────────────┼──────────────┤")
    print("  │ Tipo                       │ No superv.   │ No superv.   │ Supervisado  │")
    print("  │ Necesita falsos p/entrenar │ No           │ No           │ Sí           │")
    print("  │ Detección anomalías nuevas │ Sí           │ Sí           │ Parcialmente │")
    print("  │ Métrica de anomalía        │ Error QE     │ Error recon. │ Probabilidad │")
    print(f"  │ AUC-ROC                    │ {roc_auc:.4f}       │ 0.7760       │ ~0.9700      │")
    print("  │ Visualización topológica   │ Sí (mapa 2D) │ No           │ No           │")
    print("  │ Interpretabilidad          │ Muy alta     │ Alta         │ Baja         │")
    print("  │ Velocidad (CPU)            │ Muy rápida   │ Muy rápida   │ Moderada     │")
    print("  │ Uso en ALI                 │ Filtro+viz.  │ Filtro       │ Decisión fin.│")
    print("  └────────────────────────────┴──────────────┴──────────────┴──────────────┘")

    print()
    print("  ANÁLISIS DE LA TOPOLOGÍA (SOM Denominación):")
    print("  El SOM revela qué denominaciones son visualmente similares.")
    print("  Si bs20 (naranja) y bs50 (morado) quedan en zonas adyacentes")
    print("  del mapa, explica por qué el Modelo A los confunde a veces.")
    print("  Zonas oscuras en la U-Matrix = fronteras entre denominaciones.")
    print()
    print("  APORTE ÚNICO DEL SOM vs RBM:")
    print("  La RBM detecta anomalías pero no explica la ESTRUCTURA del espacio.")
    print("  El SOM produce un mapa 2D navegable que muestra visualmente")
    print("  la relación entre denominaciones → útil para el informe de tesis.")

    # Guardar resumen JSON
    resumen = {
        "modelo": "MiniSom (SOM de Kohonen)",
        "som_x": SOM_X, "som_y": SOM_Y,
        "n_neuronas": SOM_X * SOM_Y,
        "n_iter": N_ITER_AUT,
        "img_size": IMG_SIZE,
        "n_train_autenticos": int(len(X_tr_aut)),
        "n_test_total": int(len(y_te)),
        "auc_roc": float(roc_auc),
        "umbral_optimo_qe": float(umbral_opt),
        "tpr_opt": float(tpr_opt),
        "fpr_opt": float(fpr_opt),
        "qe_media_autenticos": float(qe_aut.mean()),
        "qe_media_falsos": float(qe_fals.mean()),
        "separacion_qe": float(sep),
    }
    json_path = REPORTE_DIR / "18_som_resumen.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(resumen, f, indent=2)

    print()
    print("=" * 70)
    print(f"  Resumen guardado : {json_path}")
    print(f"  Figura guardada  : {ruta_fig}")
    print("=" * 70)


if __name__ == "__main__":
    main()
