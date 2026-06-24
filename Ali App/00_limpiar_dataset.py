"""
=============================================================================
SCRIPT 00 - LIMPIEZA DEL DATASET
=============================================================================
QUE HACE ESTE SCRIPT:
1. Corrige la orientacion EXIF de las fotos del celular
(resuelve el problema de "imagenes invertidas").
2. Valida que cada imagen tenga una resolucion minima razonable.
3. Detecta y descarta imagenes borrosas (varianza del Laplaciano).
4. Detecta duplicados perceptuales (phash) y conserva uno solo.
5. Normaliza los nombres de archivo (quita espacios, pasa a minusculas).
6. Genera un reporte CSV con el estado de cada imagen.

OPCIONALMENTE puede filtrar por numero de archivo, util si tu dataset
tiene mezcladas imagenes originales (001-099) con augmentadas (>=100).

ENTRADA:  dataset/raw/{clase}/{denominacion}/{cara}/*.jpg
SALIDA:   dataset/raw_limpio/{clase}/{denominacion}/{cara}/*.jpg
DESCARTE: dataset/raw_descartados/{motivo}/...
REPORTE:  dataset/reportes/00_limpieza_reporte.csv

USO:
    .venv\\Scripts\\activate
    python scripts/00_limpiar_dataset.py

    # Solo procesar imagenes originales (numero <= 99), ignorando augmentadas:
    python scripts/00_limpiar_dataset.py --solo-originales

    # Cambiar el umbral de borroso (por defecto 80):
    python scripts/00_limpiar_dataset.py --umbral-borroso 50

REQUISITOS (ya instalados por setup_entorno.bat):
    opencv-python, pillow, imagehash, tqdm, numpy
=============================================================================
"""

import argparse
import csv
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import imagehash
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
from tqdm import tqdm


# =============================================================================
# CONFIGURACION
# =============================================================================
BASE_DIR     = Path(__file__).parent.parent          # carpeta raiz del proyecto
RAW_DIR      = BASE_DIR / "raw"
LIMPIO_DIR   = BASE_DIR / "raw_limpio"
DESCARTE_DIR = BASE_DIR / "raw_descartados"
REPORTE_DIR  = BASE_DIR / "reportes"
REPORTE_CSV  = REPORTE_DIR / "00_limpieza_reporte.csv"

CLASES         = ["autentico", "falsificado"]
DENOMINACIONES = ["bs10", "bs20", "bs50", "bs100", "bs200"]
CARAS          = ["anverso", "reverso"]
EXTENSIONES    = {".jpg", ".jpeg", ".png", ".bmp"}

# Umbrales por defecto (pueden sobreescribirse por linea de comandos)
RES_MINIMA           = 480     # pixeles del lado mas corto
UMBRAL_BORROSO       = 80.0    # < 80 se considera borrosa (foto de celular movida)
UMBRAL_HASH          = 5       # distancia Hamming <= 5 -> mismo billete
MAX_NUMERO_ORIGINAL  = 99      # si --solo-originales, solo procesa numeros <= esto


# =============================================================================
# FUNCIONES AUXILIARES
# =============================================================================
def normalizar_nombre(nombre: str) -> str:
    """
    Limpia y normaliza el nombre de un archivo.
    Ejemplo: 'bs50_auth_rev_001 .JPG' -> 'bs50_auth_rev_001.jpg'
    """
    base = nombre.strip().replace(" ", "")
    p = Path(base)
    return p.stem.lower() + p.suffix.lower().replace(".jpeg", ".jpg")


def extraer_numero(stem: str) -> int | None:
    """Extrae el numero final del nombre. bs10_auth_anv_017 -> 17"""
    for parte in reversed(stem.split("_")):
        if parte.isdigit():
            return int(parte)
    return None


def corregir_exif(ruta: Path) -> Image.Image | None:
    """
    Carga una imagen aplicando la correccion EXIF de orientacion.
    Devuelve un objeto PIL.Image o None si falla.
    """
    try:
        img = Image.open(ruta)
        # exif_transpose lee el tag de orientacion y rota fisicamente los pixeles
        img = ImageOps.exif_transpose(img)
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img
    except (UnidentifiedImageError, OSError) as e:
        print(f"  [!] No se pudo abrir {ruta.name}: {e}")
        return None


def calcular_blur(img_pil: Image.Image) -> float:
    """
    Devuelve la varianza del Laplaciano. Valores bajos = imagen borrosa.
    Como referencia: enfocada >150, levemente movida 80-150, borrosa <80.
    """
    arr = np.array(img_pil.convert("L"))
    return float(cv2.Laplacian(arr, cv2.CV_64F).var())


def calcular_phash(img_pil: Image.Image) -> imagehash.ImageHash:
    """Hash perceptual (64 bits) para deteccion de duplicados."""
    return imagehash.phash(img_pil)


def guardar_imagen(img_pil: Image.Image, destino: Path) -> None:
    """Guarda la imagen como JPG calidad 95."""
    destino.parent.mkdir(parents=True, exist_ok=True)
    img_pil.save(destino, format="JPEG", quality=95, optimize=True)


def copiar_descarte(origen: Path, motivo: str, nombre: str) -> None:
    """Copia la imagen original a la carpeta de descartes."""
    destino = DESCARTE_DIR / motivo / nombre
    destino.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(origen, destino)


# =============================================================================
# RECOLECCION
# =============================================================================
def recolectar_imagenes(solo_originales: bool) -> list[dict]:
    """Recorre la estructura raw/ y devuelve la lista de imagenes a procesar."""
    tareas = []
    for clase in CLASES:
        for denom in DENOMINACIONES:
            for cara in CARAS:
                carpeta = RAW_DIR / clase / denom / cara
                if not carpeta.exists():
                    continue
                for archivo in sorted(carpeta.iterdir()):
                    if archivo.suffix.lower() not in EXTENSIONES:
                        continue
                    if solo_originales:
                        num = extraer_numero(archivo.stem)
                        if num is not None and num > MAX_NUMERO_ORIGINAL:
                            continue
                    tareas.append({
                        "ruta"        : archivo,
                        "clase"       : clase,
                        "denominacion": denom,
                        "cara"        : cara,
                    })
    return tareas


# =============================================================================
# MAIN
# =============================================================================
def main() -> None:
    parser = argparse.ArgumentParser(description="Limpieza pre-procesamiento del dataset")
    parser.add_argument("--solo-originales", action="store_true",
                        help=f"Procesar solo imagenes con numero <= {MAX_NUMERO_ORIGINAL}")
    parser.add_argument("--umbral-borroso", type=float, default=UMBRAL_BORROSO,
                        help=f"Umbral de varianza Laplaciana (default {UMBRAL_BORROSO})")
    parser.add_argument("--umbral-hash", type=int, default=UMBRAL_HASH,
                        help=f"Distancia Hamming maxima para duplicados (default {UMBRAL_HASH})")
    parser.add_argument("--res-minima", type=int, default=RES_MINIMA,
                        help=f"Resolucion minima del lado corto (default {RES_MINIMA})")
    args = parser.parse_args()

    umbral_borroso = args.umbral_borroso
    umbral_hash    = args.umbral_hash
    res_minima     = args.res_minima

    print("=" * 65)
    print("  SCRIPT 00 - LIMPIEZA DEL DATASET")
    print("  Correccion EXIF + deteccion de borrosas/duplicadas")
    print("=" * 65)

    if not RAW_DIR.exists():
        print(f"\n[ERROR] No existe la carpeta: {RAW_DIR}")
        sys.exit(1)

    LIMPIO_DIR.mkdir(parents=True, exist_ok=True)
    DESCARTE_DIR.mkdir(parents=True, exist_ok=True)
    REPORTE_DIR.mkdir(parents=True, exist_ok=True)

    tareas = recolectar_imagenes(args.solo_originales)
    if not tareas:
        print("\n[!] No se encontraron imagenes para procesar.")
        sys.exit(0)

    print(f"\n  Imagenes a procesar  : {len(tareas)}")
    print(f"  Carpeta limpia       : {LIMPIO_DIR}")
    print(f"  Carpeta descartes    : {DESCARTE_DIR}")
    print(f"  Solo originales      : {'si' if args.solo_originales else 'no (todas)'}")
    print(f"  Resolucion minima    : {res_minima}px (lado corto)")
    print(f"  Umbral borroso       : {umbral_borroso}")
    print(f"  Umbral duplicado     : Hamming <= {umbral_hash}")
    print()

    registros = []
    hashes_vistos: list[tuple[imagehash.ImageHash, str]] = []
    contador = defaultdict(int)

    print("Procesando imagenes...")
    for t in tqdm(tareas, unit="img"):
        ruta_orig   = t["ruta"]
        nombre_norm = normalizar_nombre(ruta_orig.name)

        registro = {
            "archivo_original" : ruta_orig.name,
            "archivo_final"    : nombre_norm,
            "clase"            : t["clase"],
            "denominacion"     : t["denominacion"],
            "cara"             : t["cara"],
            "estado"           : "",
            "motivo"           : "",
            "resolucion"       : "",
            "blur_score"       : "",
            "phash"            : "",
        }

        # 1) abrir y corregir EXIF
        img = corregir_exif(ruta_orig)
        if img is None:
            registro["estado"] = "descartado"
            registro["motivo"] = "no_legible"
            copiar_descarte(ruta_orig, "no_legibles", ruta_orig.name)
            registros.append(registro)
            contador["no_legible"] += 1
            continue

        registro["resolucion"] = f"{img.width}x{img.height}"

        # 2) resolucion minima
        if min(img.width, img.height) < res_minima:
            registro["estado"] = "descartado"
            registro["motivo"] = "baja_resolucion"
            copiar_descarte(ruta_orig, "baja_resolucion", ruta_orig.name)
            registros.append(registro)
            contador["baja_resolucion"] += 1
            continue

        # 3) blur
        blur = calcular_blur(img)
        registro["blur_score"] = f"{blur:.1f}"
        if blur < umbral_borroso:
            registro["estado"] = "descartado"
            registro["motivo"] = "borrosa"
            copiar_descarte(ruta_orig, "borrosas", ruta_orig.name)
            registros.append(registro)
            contador["borrosa"] += 1
            continue

        # 4) duplicados perceptuales
        h = calcular_phash(img)
        registro["phash"] = str(h)
        es_duplicada = False
        for h_visto, nombre_visto in hashes_vistos:
            if (h - h_visto) <= umbral_hash:
                registro["estado"] = "descartado"
                registro["motivo"] = f"duplicada_de:{nombre_visto}"
                copiar_descarte(ruta_orig, "duplicadas", ruta_orig.name)
                contador["duplicada"] += 1
                es_duplicada = True
                break
        if es_duplicada:
            registros.append(registro)
            continue

        hashes_vistos.append((h, nombre_norm))

        # 5) guardar imagen limpia
        destino = LIMPIO_DIR / t["clase"] / t["denominacion"] / t["cara"] / nombre_norm
        guardar_imagen(img, destino)
        registro["estado"] = "ok"
        registros.append(registro)
        contador["ok"] += 1

    # ---- escribir reporte CSV ----
    if registros:
        campos = list(registros[0].keys())
        with open(REPORTE_CSV, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=campos)
            w.writeheader()
            w.writerows(registros)

    # ---- resumen ----
    print("\n" + "=" * 65)
    print("  RESUMEN DE LA LIMPIEZA")
    print("=" * 65)
    total = len(registros)
    print(f"  Total procesadas       : {total}")
    print(f"  Aceptadas              : {contador['ok']}")
    print(f"  Descartadas (borrosas) : {contador['borrosa']}")
    print(f"  Descartadas (duplicad.): {contador['duplicada']}")
    print(f"  Descartadas (baja res.): {contador['baja_resolucion']}")
    print(f"  No legibles            : {contador['no_legible']}")
    print()
    print(f"  Reporte detallado: {REPORTE_CSV}")
    print()

    # distribucion en carpeta limpia
    print("  Distribucion en raw_limpio/ :")
    dist = defaultdict(int)
    for r in registros:
        if r["estado"] == "ok":
            key = f"{r['clase']}/{r['denominacion']}/{r['cara']}"
            dist[key] += 1
    if dist:
        for k in sorted(dist.keys()):
            print(f"    {k:40s} {dist[k]:4d}")
    else:
        print("    (vacia: todas las imagenes fueron descartadas)")

    print()
    print("=" * 65)
    print("  Proximos pasos:")
    print("  1. Revisa raw_descartados/ por si quieres rescatar alguna.")
    print("  2. Abre reportes/00_limpieza_reporte.csv en Excel para auditar.")
    print("  3. Si todo se ve bien, sigue al siguiente paso.")
    print("=" * 65)


if __name__ == "__main__":
    main()
