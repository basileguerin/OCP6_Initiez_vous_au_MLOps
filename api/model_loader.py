"""
Chargement unique du modèle et des métadonnées de production.
Ce module est importé au démarrage de l'API — jamais à chaque requête.

Stratégie d'inférence (par ordre de priorité) :
  1. ONNX Runtime  — si model_reduit.onnx est présent (plus rapide)
  2. LightGBM pkl  — fallback si ONNX non disponible
"""
import json
import pickle
from pathlib import Path

import numpy as np

# Variables globales — initialisées une seule fois par load_model()
model    = None          # modèle LightGBM pkl (fallback uniquement)
FEATURES: list[str] = []
SEUIL: float = 0.53

_MODELS_DIR = Path(__file__).parent.parent / "models"
_ONNX_PATH  = _MODELS_DIR / "model_reduit.onnx"

# Session ONNX Runtime — None si non disponible
_sess       = None
_input_name = None
_proba_name = None


def load_model() -> None:
    """
    Charge le modèle au démarrage de l'API.
    Priorité : ONNX Runtime > pkl LightGBM.
    """
    global model, FEATURES, SEUIL, _sess, _input_name, _proba_name

    features_path = _MODELS_DIR / "features_selectionnees.json"
    if not features_path.exists():
        raise FileNotFoundError(f"Features introuvables : {features_path}")

    with open(features_path) as f:
        meta = json.load(f)
        FEATURES = meta["features"]
        SEUIL = 0.53  # seuil métier — ne pas utiliser 0.5

    # Tentative ONNX Runtime (v2 — runtime C++ compilé)
    if _ONNX_PATH.exists():
        try:
            import onnxruntime as rt
            _sess       = rt.InferenceSession(str(_ONNX_PATH), providers=["CPUExecutionProvider"])
            _input_name = _sess.get_inputs()[0].name
            _proba_name = _sess.get_outputs()[1].name
            print(f"[model_loader] ONNX Runtime chargé ({_ONNX_PATH.name}), seuil = {SEUIL}")
            return
        except Exception as e:
            print(f"[model_loader] ONNX non disponible ({e}), fallback pkl")

    # Fallback pkl LightGBM
    model_path = _MODELS_DIR / "model_reduit.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Modèle introuvable : {model_path}")
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    print(f"[model_loader] pkl chargé ({model_path.name}), seuil = {SEUIL}")


def predict_score(feature_values: list[float]) -> float:
    """
    Retourne le score de défaut (probabilité entre 0 et 1).
    feature_values : liste ordonnée selon FEATURES (20 valeurs float).
    """
    if _sess is not None:
        # ONNX Runtime attend float32, forme (1, 20)
        arr = np.array([feature_values], dtype=np.float32)
        return float(_sess.run([_proba_name], {_input_name: arr})[0][0][1])
    # Fallback pkl — DataFrame nommé pour éviter les warnings LightGBM
    import pandas as pd
    df = pd.DataFrame([feature_values], columns=FEATURES)
    return float(model.predict_proba(df)[0][1])


def is_loaded() -> bool:
    return model is not None or _sess is not None


def inference_mode() -> str:
    return "onnx" if _sess is not None else "pkl"
