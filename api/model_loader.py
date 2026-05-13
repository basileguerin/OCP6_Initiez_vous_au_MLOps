"""
Chargement unique du modèle et des métadonnées de production.
Ce module est importé au démarrage de l'API — jamais à chaque requête.
"""
import pickle
import json
from pathlib import Path

# Variables globales — initialisées une seule fois par load_model()
model = None
FEATURES: list[str] = []
SEUIL: float = 0.53

# Chemin vers les artefacts (depuis la racine du projet)
_MODELS_DIR = Path(__file__).parent.parent / "models"


def load_model() -> None:
    """Charge le modèle pkl et la liste ordonnée des 20 features depuis le disque."""
    global model, FEATURES, SEUIL

    model_path = _MODELS_DIR / "model_reduit.pkl"
    features_path = _MODELS_DIR / "features_selectionnees.json"

    if not model_path.exists():
        raise FileNotFoundError(f"Modèle introuvable : {model_path}")
    if not features_path.exists():
        raise FileNotFoundError(f"Features introuvables : {features_path}")

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    with open(features_path) as f:
        meta = json.load(f)
        FEATURES = meta["features"]
        # Seuil métier figé à 0.53 — optimisé sur coût FN=10×FP (notebook 06)
        SEUIL = 0.53

    print(f"[model_loader] Modèle chargé : {len(FEATURES)} features, seuil = {SEUIL}")
