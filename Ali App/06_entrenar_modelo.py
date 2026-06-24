"""
=============================================================================
SCRIPT 06 - ENTRENAMIENTO DEL MODELO DE CLASIFICACION MULTICLASE
=============================================================================
QUE HACE ESTE SCRIPT:
  Entrena un clasificador multiclase de billetes bolivianos por denominacion
  mediante Transfer Learning + Fine-Tuning sobre MobileNetV2 (default)
  o EfficientNet-B0, preentrenadas en ImageNet.

  VERSION 2 (mejora de generalizacion):
    Tras detectar que el modelo fallaba en condiciones reales de uso pese a
    un buen accuracy en test (problema de "domain shift" / sobreajuste a las
    condiciones controladas del dataset), se reforzo:
      - Data augmentation mucho mas agresiva (brillo, contraste, rotacion,
        zoom, ruido) para simular luz artificial, penumbra, angulos y fondos
        diversos del uso real.
      - Mayor regularizacion (dropout 0.5/0.4 + L2 en la capa densa).
      - class_weight para compensar el desbalance leve entre clases.
      - 5 clases (solo denominacion) en vez de 10: unifica anverso/reverso,
        duplica las imagenes por clase y hace el modelo mas robusto.
    Esta version requiere ademas mas datos de entrenamiento tomados en
    condiciones variadas (ver PROTOCOLO_RECOLECCION_DATOS.md).

  Flujo:
    FASE 1 - Transfer Learning
      Congela toda la red preentrenada y entrena solo el cabezal nuevo.
      Aprende a mapear las features de ImageNet a las 5 clases de billetes.

    FASE 2 - Fine-Tuning
      Descongela las ultimas N capas y reentrena con learning rate bajo.
      Ajusta las capas finales del backbone a las caracteristicas del billete.

ENTRADA:
  dataset_clasificacion/{train,valid,test}/{clase}/*.jpg

SALIDA:
  models/
    mobilenet_billetes.keras           (modelo entrenado)
    mobilenet_billetes_history.json    (curvas de loss y accuracy)
  reportes/
    06_entrenamiento_log.txt           (log completo del entrenamiento)

USO:
  .venv\\Scripts\\activate
  python scripts/06_entrenar_modelo.py                   # MobileNetV2 default
  python scripts/06_entrenar_modelo.py --arq efficient   # EfficientNet-B0
  python scripts/06_entrenar_modelo.py --epocas-tl 20 --epocas-ft 15
  python scripts/06_entrenar_modelo.py --batch 16        # si te quedas sin GPU memory

REQUISITOS:
  tensorflow (con CUDA), numpy, pillow
=============================================================================
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks


# =============================================================================
# CONFIGURACION
# =============================================================================
BASE_DIR     = Path(__file__).parent.parent
DATASET_DIR  = BASE_DIR / "dataset_clasificacion"
MODELS_DIR   = BASE_DIR / "models"
REPORTE_DIR  = BASE_DIR / "reportes"

# Clases detectadas automaticamente desde las subcarpetas de train/.
# Orden fijo para reproducibilidad: las denominaciones primero (orden numerico),
# "desconocido" al final si existe.
# Al agregar el script 09 (clase desconocido) se pasa automaticamente a 6 clases.
def _detectar_clases(dataset_dir: Path) -> list[str]:
    train_dir = dataset_dir / "train"
    if not train_dir.exists():
        # fallback a las 5 clases originales si todavia no existe el dataset
        return ["bs10", "bs20", "bs50", "bs100", "bs200"]
    denominaciones = sorted(
        [d.name for d in train_dir.iterdir()
         if d.is_dir() and d.name.startswith("bs")],
        key=lambda s: int(s.replace("bs", ""))  # orden numerico: 10,20,50,100,200
    )
    otras = sorted(
        [d.name for d in train_dir.iterdir()
         if d.is_dir() and not d.name.startswith("bs")]
    )
    return denominaciones + otras  # ej: ["bs10","bs20","bs50","bs100","bs200","desconocido"]

CLASES = _detectar_clases(Path(__file__).parent.parent / "dataset_clasificacion")


# =============================================================================
# DATA AUGMENTATION AGRESIVA (sin volteos para no invertir texto)
# =============================================================================
def construir_augmentacion():
    """
    Capas de augmentation aplicadas en linea durante el entrenamiento.

    OBJETIVO: simular la variabilidad de las condiciones REALES de uso de la
    app (luz artificial, penumbra, fondos diversos, angulos y distancias) que
    el dataset original — tomado en condiciones controladas — no cubre.
    Cada epoca el modelo ve las mismas fotos pero alteradas al azar, lo que
    lo obliga a aprender el billete en si y no las condiciones de la foto.

    No se usan volteos (flip) porque invertirian el texto del billete.
    """
    # Augmentation MODERADA (v3): mas fuerte que v1 (que sobreajustaba) pero
    # menos agresiva que v2 (que hacia colapsar al modelo en la clase mayoritaria
    # con tan pocas imagenes). Buscamos un balance que permita aprender los
    # patrones del billete sin destruirlos durante el entrenamiento.
    return tf.keras.Sequential([
        layers.RandomRotation(0.07),                          # +/-25 grados aprox.
        layers.RandomTranslation(0.10, 0.10),                 # +/-10% desplazamiento
        layers.RandomZoom(0.15),                              # zoom in/out +/-15%
        layers.RandomBrightness(0.25, value_range=(0, 255)),  # luz variable, sin extremos
        layers.RandomContrast(0.25),                          # contraste variable, sin extremos
    ], name="augmentacion")


# =============================================================================
# MODELO
# =============================================================================
def construir_modelo(arquitectura, n_clases, img_size):
    """Construye backbone preentrenado + cabezal nuevo."""
    if arquitectura == "mobilenet":
        base = tf.keras.applications.MobileNetV2(
            input_shape=(img_size, img_size, 3),
            include_top=False,
            weights="imagenet",
        )
        preprocesar = tf.keras.applications.mobilenet_v2.preprocess_input
    elif arquitectura == "efficient":
        base = tf.keras.applications.EfficientNetB0(
            input_shape=(img_size, img_size, 3),
            include_top=False,
            weights="imagenet",
        )
        preprocesar = tf.keras.applications.efficientnet.preprocess_input
    else:
        raise ValueError(f"Arquitectura desconocida: {arquitectura}")

    base.trainable = False  # FASE 1: congelado

    aug = construir_augmentacion()

    inputs = layers.Input(shape=(img_size, img_size, 3))
    x = aug(inputs)
    x = preprocesar(x)
    # Ruido gaussiano leve sobre el rango normalizado [-1, 1]
    x = layers.GaussianNoise(0.03)(x)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    # Regularizacion MODERADA: dropout suficiente para evitar sobreajuste
    # pero sin matar la capacidad de aprender (sin L2 esta vez).
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(n_clases, activation="softmax")(x)

    modelo = models.Model(inputs, outputs, name=f"{arquitectura}_billetes")
    return modelo, base


# =============================================================================
# CARGA DE DATASETS
# =============================================================================
def cargar_datasets(img_size, batch_size):
    """Carga los tres splits desde dataset_clasificacion/."""
    common = dict(
        labels="inferred",
        label_mode="int",
        class_names=CLASES,
        color_mode="rgb",
        batch_size=batch_size,
        image_size=(img_size, img_size),
        shuffle=True,
        seed=42,
    )
    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_DIR / "train", **common,
    )
    valid_ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_DIR / "valid", **{**common, "shuffle": False},
    )
    test_ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_DIR / "test", **{**common, "shuffle": False},
    )

    # cache + prefetch para acelerar
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(AUTOTUNE)
    valid_ds = valid_ds.cache().prefetch(AUTOTUNE)
    test_ds = test_ds.cache().prefetch(AUTOTUNE)

    return train_ds, valid_ds, test_ds


# =============================================================================
# MAIN
# =============================================================================
def main():
    p = argparse.ArgumentParser(description="Entrena clasificador de billetes")
    p.add_argument("--arq", choices=["mobilenet", "efficient"], default="mobilenet",
                   help="Arquitectura: mobilenet (default) o efficient")
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--batch", type=int, default=32, help="Batch size (default 32)")
    p.add_argument("--epocas-tl", type=int, default=40,
                   help="Epocas de Transfer Learning (default 40)")
    p.add_argument("--epocas-ft", type=int, default=30,
                   help="Epocas de Fine-Tuning (default 30)")
    p.add_argument("--lr-tl", type=float, default=1e-3,
                   help="Learning rate Transfer Learning (default 1e-3)")
    p.add_argument("--lr-ft", type=float, default=1e-5,
                   help="Learning rate Fine-Tuning (default 1e-5)")
    p.add_argument("--capas-ft", type=int, default=20,
                   help="Numero de capas finales a descongelar en FT (default 20)")
    p.add_argument("--paciencia", type=int, default=8,
                   help="EarlyStopping patience (default 8)")
    args = p.parse_args()

    # verificaciones iniciales
    if not DATASET_DIR.exists():
        print(f"ERROR: no existe {DATASET_DIR}")
        print("Corre primero: python scripts/04_preparar_clasificacion.py --clean --max-por-billete 12")
        return 1

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTE_DIR.mkdir(parents=True, exist_ok=True)
    log_path = REPORTE_DIR / "06_entrenamiento_log.txt"

    log_lines = []
    def log(msg=""):
        print(msg)
        log_lines.append(msg)

    log("=" * 70)
    log(f"  SCRIPT 06 - ENTRENAMIENTO ({args.arq.upper()})")
    log("=" * 70)
    log(f"  Inicio          : {datetime.now().isoformat(timespec='seconds')}")
    log(f"  Arquitectura    : {args.arq}")
    log(f"  Imagen          : {args.img_size}x{args.img_size}")
    log(f"  Batch size      : {args.batch}")
    log(f"  Epocas TL       : {args.epocas_tl}")
    log(f"  Epocas FT       : {args.epocas_ft}")
    log(f"  LR TL           : {args.lr_tl}")
    log(f"  LR FT           : {args.lr_ft}")
    log(f"  Capas a descongelar en FT: {args.capas_ft}")
    log()

    # GPU info
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for g in gpus:
            log(f"  GPU detectada: {g.name}")
        # Permitir crecimiento de memoria, evita reservar toda la VRAM
        for g in gpus:
            try:
                tf.config.experimental.set_memory_growth(g, True)
            except Exception:
                pass
    else:
        log("  [!] No se detecto GPU. El entrenamiento sera lento.")
    log()

    # =========================================================================
    # CARGAR DATASETS
    # =========================================================================
    log("Cargando datasets...")
    train_ds, valid_ds, test_ds = cargar_datasets(args.img_size, args.batch)

    # Contar imagenes por clase en train (para n_train y class_weight)
    from collections import Counter
    conteo_clases = Counter()
    for _, label in train_ds.unbatch():
        conteo_clases[int(label.numpy())] += 1
    n_train = sum(conteo_clases.values())
    n_valid = sum(1 for _ in valid_ds.unbatch())
    n_test  = sum(1 for _ in test_ds.unbatch())
    log(f"  Train : {n_train} imagenes")
    log(f"  Valid : {n_valid} imagenes")
    log(f"  Test  : {n_test} imagenes")
    log(f"  Clases: {len(CLASES)}")
    log()

    # class_weight: compensa el desbalance leve entre clases. Las clases con
    # menos fotos pesan mas en la perdida, evitando que el modelo las ignore.
    class_weight = {
        i: n_train / (len(CLASES) * max(1, conteo_clases.get(i, 0)))
        for i in range(len(CLASES))
    }
    log("Pesos por clase (class_weight):")
    for i, clase in enumerate(CLASES):
        log(f"  {clase:<12} n={conteo_clases.get(i, 0):>3}  peso={class_weight[i]:.2f}")
    log()

    # =========================================================================
    # CONSTRUIR MODELO
    # =========================================================================
    log(f"Construyendo modelo {args.arq}...")
    modelo, base = construir_modelo(args.arq, len(CLASES), args.img_size)
    log(f"  Parametros totales    : {modelo.count_params():,}")
    log(f"  Parametros entrenables: {sum(tf.size(v).numpy() for v in modelo.trainable_variables):,}")
    log()

    # =========================================================================
    # FASE 1 - TRANSFER LEARNING
    # =========================================================================
    log("=" * 70)
    log("  FASE 1 - TRANSFER LEARNING (backbone congelado)")
    log("=" * 70)

    modelo.compile(
        optimizer=optimizers.Adam(args.lr_tl),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks_tl = [
        callbacks.EarlyStopping(
            monitor="val_accuracy", patience=args.paciencia,
            restore_best_weights=True, verbose=1,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1,
        ),
    ]

    historia_tl = modelo.fit(
        train_ds,
        validation_data=valid_ds,
        epochs=args.epocas_tl,
        callbacks=callbacks_tl,
        class_weight=class_weight,
        verbose=2,
    )
    log()
    log(f"  TL acc final (val): {historia_tl.history['val_accuracy'][-1]:.4f}")

    # =========================================================================
    # FASE 2 - FINE-TUNING
    # =========================================================================
    log()
    log("=" * 70)
    log(f"  FASE 2 - FINE-TUNING (descongelando ultimas {args.capas_ft} capas)")
    log("=" * 70)

    base.trainable = True
    # congelar todas excepto las ultimas N
    for capa in base.layers[:-args.capas_ft]:
        capa.trainable = False

    modelo.compile(
        optimizer=optimizers.Adam(args.lr_ft),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    log(f"  Parametros entrenables ahora: {sum(tf.size(v).numpy() for v in modelo.trainable_variables):,}")
    log()

    callbacks_ft = [
        callbacks.EarlyStopping(
            monitor="val_accuracy", patience=args.paciencia,
            restore_best_weights=True, verbose=1,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7, verbose=1,
        ),
    ]

    historia_ft = modelo.fit(
        train_ds,
        validation_data=valid_ds,
        epochs=args.epocas_ft,
        callbacks=callbacks_ft,
        class_weight=class_weight,
        verbose=2,
    )
    log()
    log(f"  FT acc final (val): {historia_ft.history['val_accuracy'][-1]:.4f}")

    # =========================================================================
    # EVALUACION SOBRE TEST
    # =========================================================================
    log()
    log("=" * 70)
    log("  EVALUACION SOBRE CONJUNTO DE PRUEBA")
    log("=" * 70)
    test_loss, test_acc = modelo.evaluate(test_ds, verbose=0)
    log(f"  Test loss     : {test_loss:.4f}")
    log(f"  Test accuracy : {test_acc:.4f}  ({test_acc * 100:.2f}%)")

    # =========================================================================
    # GUARDAR MODELO E HISTORIAL
    # =========================================================================
    modelo_path = MODELS_DIR / f"{args.arq}_billetes.keras"
    modelo.save(modelo_path)
    log()
    log(f"  Modelo guardado: {modelo_path}")

    historia_completa = {
        "fase_tl": {k: [float(v) for v in vals]
                    for k, vals in historia_tl.history.items()},
        "fase_ft": {k: [float(v) for v in vals]
                    for k, vals in historia_ft.history.items()},
        "test_loss": float(test_loss),
        "test_accuracy": float(test_acc),
        "config": vars(args),
        "fecha": datetime.now().isoformat(timespec="seconds"),
    }
    hist_path = MODELS_DIR / f"{args.arq}_billetes_history.json"
    with open(hist_path, "w", encoding="utf-8") as f:
        json.dump(historia_completa, f, indent=2)
    log(f"  Historial guardado: {hist_path}")

    log()
    log(f"  Fin: {datetime.now().isoformat(timespec='seconds')}")
    log("=" * 70)
    log("PROXIMO PASO: python scripts/07_evaluar_modelo.py")

    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
