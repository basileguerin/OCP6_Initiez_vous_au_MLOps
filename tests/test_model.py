"""
Tests unitaires du modèle de production — chargement et prédiction.

Ces tests vérifient que le modèle et ses métadonnées sont
exploitables indépendamment de l'API.
"""
import json
import pickle
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

MODELS_DIR = Path(__file__).parent.parent / "models"


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def model():
    """Charge le modèle pkl une seule fois pour tout le module."""
    with open(MODELS_DIR / "model_reduit.pkl", "rb") as f:
        return pickle.load(f)


@pytest.fixture(scope="module")
def features_meta():
    """Charge les métadonnées des features (liste + seuil)."""
    with open(MODELS_DIR / "features_selectionnees.json") as f:
        return json.load(f)


# ─── Tests de chargement ──────────────────────────────────────────────────────

class TestChargement:
    def test_fichier_modele_existe(self):
        """Le fichier model_reduit.pkl doit exister dans models/."""
        assert (MODELS_DIR / "model_reduit.pkl").exists()

    def test_fichier_onnx_existe(self):
        """Le fichier model_reduit.onnx doit exister après conversion."""
        assert (MODELS_DIR / "model_reduit.onnx").exists()

    def test_fichier_features_existe(self):
        """Le fichier features_selectionnees.json doit exister dans models/."""
        assert (MODELS_DIR / "features_selectionnees.json").exists()

    def test_modele_charge_sans_erreur(self, model):
        """Le chargement du modèle ne doit pas lever d'exception."""
        assert model is not None

    def test_modele_a_methode_predict_proba(self, model):
        """Le modèle doit exposer predict_proba (classification probabiliste)."""
        assert hasattr(model, "predict_proba")

    def test_features_contient_20_elements(self, features_meta):
        """Le modèle de production doit utiliser exactement 20 features."""
        assert features_meta["k"] == 20
        assert len(features_meta["features"]) == 20

    def test_features_contient_ext_source_2(self, features_meta):
        """EXT_SOURCE_2 est la feature SHAP la plus importante — doit être présente."""
        assert "EXT_SOURCE_2" in features_meta["features"]


# ─── Tests de prédiction ──────────────────────────────────────────────────────

class TestPrediction:
    def _vecteur(self, features_meta, valeurs: dict) -> pd.DataFrame:
        """Construit un DataFrame avec les noms de colonnes attendus par le modèle."""
        features = features_meta["features"]
        return pd.DataFrame([[valeurs[f] for f in features]], columns=features)

    def test_predict_proba_retourne_deux_colonnes(self, model, features_meta):
        """predict_proba doit retourner une matrice (n, 2) : [proba_0, proba_1]."""
        vecteur = self._vecteur(features_meta, {f: 0.5 for f in features_meta["features"]})
        result = model.predict_proba(vecteur)
        assert result.shape == (1, 2)

    def test_proba_est_entre_0_et_1(self, model, features_meta):
        """Les probabilités prédites sont toujours entre 0 et 1."""
        vecteur = self._vecteur(features_meta, {f: 0.5 for f in features_meta["features"]})
        proba = model.predict_proba(vecteur)[0][1]
        assert 0.0 <= proba <= 1.0

    def test_proba_somme_a_1(self, model, features_meta):
        """Les probabilités des deux classes doivent sommer à 1 (± epsilon)."""
        vecteur = self._vecteur(features_meta, {f: 0.5 for f in features_meta["features"]})
        probas = model.predict_proba(vecteur)[0]
        assert abs(probas.sum() - 1.0) < 1e-6

    def test_bon_client_a_faible_score(self, model, features_meta):
        """Un profil à très faible risque (EXT_SOURCE élevés) doit avoir score < 0.5."""
        valeurs_bon = {f: 0.5 for f in features_meta["features"]}
        valeurs_bon.update({"EXT_SOURCE_2": 0.9, "EXT_SOURCE_3": 0.85, "EXT_SOURCE_1": 0.80})
        vecteur = self._vecteur(features_meta, valeurs_bon)
        score = model.predict_proba(vecteur)[0][1]
        assert score < 0.5

    def test_predict_score_onnx_retourne_float(self):
        """predict_score() via ONNX Runtime retourne un float entre 0 et 1."""
        import api.model_loader as ml
        ml.load_model()
        score = ml.predict_score([0.5] * 20)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_predict_score_fallback_pkl(self, model, features_meta):
        """predict_score() utilise le pkl quand _sess est None (fallback)."""
        import api.model_loader as ml
        with patch.object(ml, "_sess", None), \
             patch.object(ml, "model", model), \
             patch.object(ml, "FEATURES", features_meta["features"]):
            score = ml.predict_score([0.5] * 20)
        assert 0.0 <= score <= 1.0

    def test_load_model_fallback_pkl(self):
        """load_model() charge le pkl si le fichier ONNX est absent."""
        import api.model_loader as ml
        original_onnx = ml._ONNX_PATH
        try:
            ml._ONNX_PATH = MODELS_DIR / "nonexistent.onnx"
            ml._sess = None
            ml.model = None
            ml.load_model()
            assert ml.model is not None
            assert ml._sess is None
        finally:
            ml._ONNX_PATH = original_onnx
            ml.load_model()  # restaure ONNX pour les autres tests

    def test_load_model_erreur_features_manquantes(self):
        """load_model() lève FileNotFoundError si features_selectionnees.json est absent."""
        import api.model_loader as ml
        original = ml._MODELS_DIR
        try:
            ml._MODELS_DIR = Path("/tmp/inexistant")
            with pytest.raises(FileNotFoundError):
                ml.load_model()
        finally:
            ml._MODELS_DIR = original
            ml.load_model()

    def test_mauvais_client_a_score_eleve(self, model, features_meta):
        """Un profil à très fort risque (EXT_SOURCE très bas) doit avoir score > 0.5."""
        valeurs_risque = {f: 0.5 for f in features_meta["features"]}
        valeurs_risque.update({
            "EXT_SOURCE_2": 0.02, "EXT_SOURCE_3": 0.03, "EXT_SOURCE_1": 0.04,
            "INSTALL_MANQUE_MOYEN": 8000.0, "PREV_NB_REFUSEES": 5.0,
        })
        vecteur = self._vecteur(features_meta, valeurs_risque)
        score = model.predict_proba(vecteur)[0][1]
        assert score > 0.5
