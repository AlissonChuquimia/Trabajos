# -*- coding: utf-8 -*-
"""Alternativa: genera data/reviews.csv con un dataset REAL en espanol (muchocine).
Ejecutar en tu maquina (requiere internet):  pip install datasets pandas"""
import sys, pandas as pd

def main():
    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit("Instala:  pip install datasets pandas")
    try:
        ds = load_dataset("muchocine", split="train", trust_remote_code=True)
    except Exception as e:
        sys.exit(f"No se pudo descargar: {e}")
    df = ds.to_pandas()
    text_col = next((c for c in df.columns if df[c].dtype == object and df[c].astype(str).str.len().mean() > 30), None)
    star_col = next((c for c in df.columns if any(k in c.lower() for k in ("star", "rating", "punt", "score"))), None)
    stars = pd.to_numeric(df[star_col], errors="coerce")
    out = pd.DataFrame({"texto": df[text_col].astype(str), "_s": stars})
    out = out[out["_s"] != 3]
    out["sentimiento"] = (out["_s"] >= 4).astype(int)
    out[["texto", "sentimiento"]].dropna().to_csv("data/reviews.csv", index=False)
    print("OK -> data/reviews.csv")

if __name__ == "__main__":
    main()
