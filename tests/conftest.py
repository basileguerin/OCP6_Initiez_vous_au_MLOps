"""
Configuration partagée des tests pytest.

conftest.py est automatiquement lu par pytest.
Il contient les fixtures réutilisables par tous les fichiers de test.
"""
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# Ajout de la racine du projet au PYTHONPATH pour permettre `from api.xxx import ...`
sys.path.insert(0, str(Path(__file__).parent.parent))

from api.main import app


@pytest.fixture(scope="module")
def client():
    """
    Crée un client de test FastAPI réutilisé pour tout le module.
    scope="module" : le client (et donc le modèle) est chargé une seule fois
    pour tous les tests du fichier, ce qui accélère la suite.
    """
    with TestClient(app) as c:
        yield c


@pytest.fixture
def payload_valide():
    """Jeu de données valide représentant un bon client (faible risque de défaut)."""
    return {
        "EXT_SOURCE_2": 0.72,
        "EXT_SOURCE_3": 0.65,
        "EXT_SOURCE_1": 0.55,
        "RATIO_CREDIT_BIEN": 1.05,
        "INSTALL_MANQUE_MOYEN": 0.0,
        "CODE_GENDER_M": 0.0,
        "DAYS_EMPLOYED": -3000.0,
        "AMT_ANNUITY": 15000.0,
        "FLAG_OWN_CAR": 0.0,
        "CC_SOLDE_MOYEN": 2000.0,
        "BUREAU_JOURS_MOYEN": -900.0,
        "POS_NB_MOIS": 36.0,
        "AGE": 45.0,
        "AMT_CREDIT": 200000.0,
        "BUREAU_DETTE_MOYENNE": 0.0,
        "NAME_EDUCATION_TYPE_Higher_education": 1.0,
        "PREV_NB_REFUSEES": 0.0,
        "NAME_FAMILY_STATUS_Married": 1.0,
        "DAYS_ID_PUBLISH": -1500.0,
        "BUREAU_NB_ACTIFS": 1.0,
    }
