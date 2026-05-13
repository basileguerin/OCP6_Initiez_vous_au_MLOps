# Scoring Crédit — Projet MLOps

Projet réalisé dans le cadre de la formation **AI Engineer** (OpenClassrooms).
Mise en œuvre d'une démarche MLOps complète pour un modèle de scoring crédit.

---

## Contexte métier

La société fictive **"Prêt à dépenser"** propose des crédits à la consommation à des clients avec peu d'historique bancaire. L'objectif est de prédire automatiquement la probabilité de défaut de paiement d'un client (`TARGET = 1`).

- **Données** : [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk) (Kaggle)
- **Déséquilibre** : 8% de défauts / 92% de bons clients
- **Contrainte métier** : le coût d'un faux négatif (FN) est 10x supérieur à celui d'un faux positif (FP)

---

## Structure du projet

```
├── api/
│   ├── main.py               # App FastAPI — endpoints /health et /predict
│   ├── model_loader.py       # Chargement unique du modèle au démarrage
│   └── schemas.py            # Validation des 20 features avec Pydantic
├── tests/
│   ├── conftest.py           # Fixtures pytest partagées
│   ├── test_api.py           # Tests API (cas nominaux + validation entrées)
│   └── test_model.py         # Tests chargement et prédiction du modèle
├── notebooks/
│   ├── 01_EDA.ipynb              # Exploration des données
│   ├── 02_preprocessing.ipynb    # Nettoyage, encodage, jointures, feature engineering
│   ├── 03_training_mlflow.ipynb  # Entraînement de 5 modèles avec tracking MLflow
│   ├── 04_optimisation.ipynb     # Tuning Optuna, seuil métier, serving MLflow
│   ├── 05_explicabilite.ipynb    # Explicabilité SHAP (importance globale, waterfall)
│   └── 06_feature_selection.ipynb  # Sélection de features SHAP + modèle réduit pour prod
├── models/
│   ├── model_reduit.pkl              # Modèle LightGBM production (20 features)
│   └── features_selectionnees.json   # Liste des 20 features + seuil de décision
├── data/
│   ├── raw/                  # Données brutes Kaggle (non versionnées)
│   └── processed/            # Dataset final après preprocessing
├── outputs/                  # Graphiques SHAP et courbes d'évaluation
├── .github/workflows/
│   └── tests.yml             # CI : tests automatiques sur push develop
├── requirements.txt          # Dépendances complètes (notebooks + API)
├── requirements-api.txt      # Dépendances légères pour le déploiement
└── mlruns/                   # Tracking MLflow (généré automatiquement)
```

---

## Pipeline ML

### 1. EDA (`01_EDA.ipynb`)
- Distribution de la variable cible (déséquilibre 8%/92%)
- Analyse des valeurs manquantes
- Corrélations avec TARGET : `EXT_SOURCE_1/2/3` dominent
- Distribution des variables clés (âge, ancienneté professionnelle)

### 2. Preprocessing (`02_preprocessing.ipynb`)
- Nettoyage des anomalies (`DAYS_EMPLOYED = 365243`)
- Encodage : Label Encoding (variables binaires) + One-Hot Encoding
- Jointures avec 5 tables secondaires via agrégation `groupby` + `left join`
- Suppression des colonnes à >40% de NaN, imputation par la médiane
- Feature engineering : ratios financiers (`RATIO_ANNUITE_REVENU`, `RATIO_CREDIT_REVENU`...)
- **Dataset final** : 307 511 clients × 210 features

### 3. Entraînement (`03_training_mlflow.ipynb`)
Tous les modèles sont évalués avec `StratifiedKFold(n_splits=5)` + métriques loggées dans MLflow.

| Modèle | AUC test | Sensibilité | Spécificité |
|---|---|---|---|
| Logistic Regression (+ scaling) | 0.76 | 0.69 | 0.70 |
| Random Forest | 0.74 | 0.00 | 1.00 |
| XGBoost | 0.76 | 0.62 | 0.76 |
| **LightGBM** | **0.77** | **0.69** | **0.72** |
| MLP | 0.75 | 0.03 | 1.00 |

### 4. Optimisation (`04_optimisation.ipynb`)
- **Optuna** (30 trials, 5-fold CV) → meilleurs hyperparamètres LightGBM
- **Seuil métier** optimisé en minimisant le coût FN/FP
- **AUC finale** : 0.7759
- Modèle enregistré dans le **MLflow Model Registry**
- **MLflow Serving** : API REST testée via `mlflow models serve`

### 5. Explicabilité (`05_explicabilite.ipynb`)
- **Feature importance** LightGBM (gain) — top 20
- **SHAP** : importance globale (bar) + direction des effets (beeswarm)
- **Waterfall plots** : explication individuelle sur un bon client (proba = 0.021) et un client en défaut (proba = 0.928)
- Graphiques exportés dans `outputs/shap/` et loggés comme artefacts MLflow

### 6. Sélection de features pour la production (`06_feature_selection.ipynb`)
- Ranking des features par importance SHAP globale (`mean(|SHAP value|)`) — top 20 retenus
- Boucle k = 1 à 20 avec `StratifiedKFold(5)` : AUC CV pour chaque sous-ensemble
- Courbe AUC vs k monotone croissante sans plateau → k = 20 retenu (−0.010 AUC vs modèle complet)
- Seuil de décision recalculé sur probabilités OOF : **0.53** (vs 0.54 pour le modèle complet)
- Modèle final enregistré dans le MLflow Model Registry (`scoring-credit-lgbm` v2)
- Artefacts production exportés dans `models/`

---

## API de scoring (FastAPI)

L'API expose le modèle de production via deux endpoints :

| Endpoint | Méthode | Description |
|---|---|---|
| `/health` | GET | Vérifie que l'API est opérationnelle |
| `/predict` | POST | Retourne le score de défaut et la décision |
| `/docs` | GET | Interface Swagger interactive |

### Lancer l'API en local

```bash
source .venv/bin/activate
python -m uvicorn api.main:app --reload --port 8000
```

Swagger accessible sur **`http://127.0.0.1:8000/docs`**

### Exemple de requête

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "EXT_SOURCE_2": 0.62, "EXT_SOURCE_3": 0.55, "EXT_SOURCE_1": 0.48,
    "RATIO_CREDIT_BIEN": 1.10, "INSTALL_MANQUE_MOYEN": 0.0,
    "CODE_GENDER_M": 0.0, "DAYS_EMPLOYED": -2500.0, "AMT_ANNUITY": 18000.0,
    "FLAG_OWN_CAR": 0.0, "CC_SOLDE_MOYEN": 5000.0,
    "BUREAU_JOURS_MOYEN": -800.0, "POS_NB_MOIS": 24.0, "AGE": 42.0,
    "AMT_CREDIT": 270000.0, "BUREAU_DETTE_MOYENNE": 0.0,
    "NAME_EDUCATION_TYPE_Higher_education": 0.0, "PREV_NB_REFUSEES": 0.0,
    "NAME_FAMILY_STATUS_Married": 1.0, "DAYS_ID_PUBLISH": -1500.0,
    "BUREAU_NB_ACTIFS": 1.0
  }'
```

Réponse :
```json
{"score": 0.1243, "decision": "ACCEPTE", "seuil": 0.53, "latence_ms": 4.2}
```

### Tests

```bash
python -m pytest tests/ -v --cov=api --cov-report=term-missing
```

---

## MLflow

```bash
# Lancer l'UI
mlflow ui --backend-store-uri mlruns

# Serving du modèle final
mlflow models serve -m "runs:/<run_id>/model" --port 5001 --no-conda
```

UI accessible sur `http://127.0.0.1:5000`

---

## Installation

```bash
git clone https://github.com/basileguerin/Scoring-Credit.git
cd Scoring-Credit

python -m venv .venv
source .venv/bin/activate

# Dépendances complètes (notebooks + API)
pip install -r requirements.txt

# Dépendances légères (API seule, pour Docker)
pip install -r requirements-api.txt
```

> **Données non versionnées** — les fichiers CSV sont trop volumineux pour Git.
> Télécharge les données depuis [Kaggle — Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk/data) et place-les dans `data/`.

---

## Résultats

### Modèle optimisé v1 (119 features — baseline)
- **AUC = 0.776** — **Sensibilité = 0.677** — Seuil = 0.54

### Modèle production v2 (20 features — déployé)
- **AUC = 0.768** — **Sensibilité = 0.655** — **Spécificité = 0.746** — Seuil = **0.53**
- 20 features sélectionnées par SHAP sur 119 (réduction de 83%)
- Perte de performance : −0.008 AUC, −0.022 sensibilité — jugée acceptable pour la prod

Les 20 features retenues (par ordre d'importance SHAP) :
`EXT_SOURCE_2`, `EXT_SOURCE_3`, `EXT_SOURCE_1`, `RATIO_CREDIT_BIEN`, `INSTALL_MANQUE_MOYEN`,
`CODE_GENDER_M`, `DAYS_EMPLOYED`, `AMT_ANNUITY`, `FLAG_OWN_CAR`, `CC_SOLDE_MOYEN`,
`BUREAU_JOURS_MOYEN`, `POS_NB_MOIS`, `AGE`, `AMT_CREDIT`, `BUREAU_DETTE_MOYENNE`,
`NAME_EDUCATION_TYPE_Higher_education`, `PREV_NB_REFUSEES`, `NAME_FAMILY_STATUS_Married`,
`DAYS_ID_PUBLISH`, `BUREAU_NB_ACTIFS`
