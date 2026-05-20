"""
Peuple logs/predictions.jsonl en appelant l'API sur un échantillon du dataset.

Usage : python scripts/generate_logs.py [--n 500]
"""
import argparse
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

API_URL   = "http://localhost:8000/predict"
DATA_PATH = Path(__file__).parent.parent / "data" / "processed" / "dataset_final.csv"

# Noms attendus par l'API (avec underscore)
FEATURES = [
    "EXT_SOURCE_2", "EXT_SOURCE_3", "EXT_SOURCE_1",
    "RATIO_CREDIT_BIEN", "INSTALL_MANQUE_MOYEN", "CODE_GENDER_M",
    "DAYS_EMPLOYED", "AMT_ANNUITY", "FLAG_OWN_CAR", "CC_SOLDE_MOYEN",
    "BUREAU_JOURS_MOYEN", "POS_NB_MOIS", "AGE", "AMT_CREDIT",
    "BUREAU_DETTE_MOYENNE", "NAME_EDUCATION_TYPE_Higher_education",
    "PREV_NB_REFUSEES", "NAME_FAMILY_STATUS_Married",
    "DAYS_ID_PUBLISH", "BUREAU_NB_ACTIFS",
]

# Colonnes correspondantes dans le dataset (certains noms diffèrent)
FEATURES_DATASET = {f: f for f in FEATURES}
FEATURES_DATASET["NAME_EDUCATION_TYPE_Higher_education"] = "NAME_EDUCATION_TYPE_Higher education"


def appeler_api(row: pd.Series) -> bool:
    payload = {feat: float(row[feat]) for feat in FEATURES}
    try:
        resp = httpx.post(API_URL, json=payload, timeout=10.0)
        resp.raise_for_status()
        return True
    except Exception as e:
        print(f"  Erreur : {e}")
        return False


def main(n: int) -> None:
    print("Chargement du dataset...")
    cols_dataset = list(FEATURES_DATASET.values())
    df = pd.read_csv(DATA_PATH, usecols=cols_dataset).dropna()
    df = df.rename(columns={v: k for k, v in FEATURES_DATASET.items()})
    print(f"  {len(df)} lignes disponibles, échantillonnage de {n}")

    echantillon = df.sample(n=min(n, len(df)), random_state=42)

    ok, erreurs = 0, 0
    for i, (_, row) in enumerate(echantillon.iterrows()):
        if appeler_api(row):
            ok += 1
        else:
            erreurs += 1
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{n} — OK: {ok}, erreurs: {erreurs}")

    print(f"\nTerminé : {ok} logs enregistrés, {erreurs} erreurs")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=500, help="Nombre de requêtes à envoyer")
    args = parser.parse_args()
    main(args.n)
