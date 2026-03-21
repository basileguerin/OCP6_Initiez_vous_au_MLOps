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
├── data/
│   ├── raw/                  # Données brutes Kaggle (non versionnées)
│   └── processed/            # Dataset final après preprocessing
├── notebooks/
│   ├── 01_EDA.ipynb          # Exploration des données
│   ├── 02_preprocessing.ipynb # Nettoyage, encodage, jointures, feature engineering
│   ├── 03_training_mlflow.ipynb # Entraînement de 5 modèles avec tracking MLflow
│   └── 04_optimisation.ipynb # Tuning Optuna, seuil métier, serving MLflow
├── mlruns/                   # Tracking MLflow (généré automatiquement)
└── README.md
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
git clone <repo_url>
cd projet6

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

> **Données non versionnées** — les fichiers CSV sont trop volumineux pour Git.
> Télécharge les données depuis [Kaggle — Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk/data) et place-les dans `data/`.

---

## Résultats

Le modèle final **LightGBM tuné** atteint :
- **AUC = 0.776** (objectif < 0.82 pour éviter l'overfitting)
- **Sensibilité = 0.677** — détecte 68% des vrais défauts
- **Spécificité = 0.735**
- Seuil de décision optimisé sur le coût métier (FN = 10x FP)
