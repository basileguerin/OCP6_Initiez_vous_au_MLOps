# Scoring Crédit — Projet MLOps

Projet réalisé dans le cadre de la formation **AI Engineer** (OpenClassrooms).  
MLOps end-to-end : entraînement, API, déploiement, monitoring de drift.

**Contexte métier** : "Prêt à dépenser" — prédire la probabilité de défaut de paiement d'un client (`TARGET = 1`).  
Données : [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk) (Kaggle) — 307 511 clients, 8% de défauts.  
Contrainte : un faux négatif coûte **10× plus** qu'un faux positif → seuil optimisé à **0.53**.

---

## Liens

| | URL |
|---|---|
| API FastAPI (Swagger) | https://basmoket-scoring-credit-api.hf.space/docs |
| Interface Gradio (démo) | https://huggingface.co/spaces/basmoket/scoring-credit-demo |
| Repo GitHub | https://github.com/basileguerin/Scoring-Credit |

---

## Modèle de production

- **Algorithme** : LightGBM, entraîné sur 307 511 clients
- **AUC** : 0.768 — **Sensibilité** : 0.655 — **Spécificité** : 0.746
- **Seuil de décision** : 0.53 (optimisé sur coût métier FN/FP via OOF cross-validation)
- **Features** : 20 sélectionnées par importance SHAP depuis 119 features initiales

Les 20 features (ordre d'importance SHAP décroissante) :  
`EXT_SOURCE_2`, `EXT_SOURCE_3`, `EXT_SOURCE_1`, `RATIO_CREDIT_BIEN`, `INSTALL_MANQUE_MOYEN`,
`CODE_GENDER_M`, `DAYS_EMPLOYED`, `AMT_ANNUITY`, `FLAG_OWN_CAR`, `CC_SOLDE_MOYEN`,
`BUREAU_JOURS_MOYEN`, `POS_NB_MOIS`, `AGE`, `AMT_CREDIT`, `BUREAU_DETTE_MOYENNE`,
`NAME_EDUCATION_TYPE_Higher_education`, `PREV_NB_REFUSEES`, `NAME_FAMILY_STATUS_Married`,
`DAYS_ID_PUBLISH`, `BUREAU_NB_ACTIFS`

---

## Structure du projet

```
├── api/                        # API FastAPI
│   ├── main.py                 # Endpoints /health, /predict, /
│   ├── model_loader.py         # Chargement unique du modèle au démarrage
│   ├── schemas.py              # Validation Pydantic des 20 features
│   └── logger.py               # Logging JSONL des prédictions
├── gradio_app/
│   └── app.py                  # Interface de démo (déployée sur HF Spaces)
├── monitoring/
│   └── dashboard.py            # Dashboard Streamlit (drift + métriques opérationnelles)
├── notebooks/
│   ├── 01 → 06                 # Exploration, preprocessing, entraînement, SHAP, sélection
│   └── 07_data_drift.ipynb     # Analyse de drift Evidently (référence vs production)
├── scripts/
│   └── generate_logs.py        # Génère des logs de production simulés
├── tests/
│   ├── test_api.py             # Tests FastAPI (31 tests, 95% coverage)
│   └── test_model.py           # Tests chargement et prédiction
├── models/
│   ├── model_reduit.pkl        # Modèle sérialisé (production)
│   └── features_selectionnees.json
├── logs/                       # Logs de production (predictions.jsonl, non versionné)
├── outputs/                    # Graphiques SHAP, rapport drift HTML, dashboard PNG
├── Dockerfile.api              # Image Docker de l'API
├── Dockerfile.gradio           # Image Docker de l'interface Gradio
├── docker-compose.yml          # Orchestre API (8000) + Gradio (7860)
├── deploy_api.py               # Déploiement manuel sur HF Spaces (API)
├── deploy_gradio.py            # Déploiement manuel sur HF Spaces (Gradio)
└── .github/workflows/
    └── ci-cd.yml               # CI/CD : tests → déploiement HF sur push main
```

---

## Partie 1 — Entraînement (résumé)

| Notebook | Contenu |
|---|---|
| `01` — EDA | Distribution TARGET, valeurs manquantes, corrélations |
| `02` — Preprocessing | Jointures 6 tables, feature engineering, 307k × 210 features |
| `03` — Entraînement | LogReg, RandomForest, XGBoost, **LightGBM** (AUC 0.77), MLP — tracking MLflow |
| `04` — Optimisation | Optuna 30 trials, seuil métier, MLflow Model Registry + serving |
| `05` — Explicabilité | SHAP global (beeswarm) + local (waterfall) |
| `06` — Sélection features | Top 20 SHAP → modèle production (AUC 0.768, seuil 0.53) |

---

## Partie 2 — Production

### Lancer l'API en local

```bash
git clone https://github.com/basileguerin/Scoring-Credit.git
cd Scoring-Credit
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-api.txt
python -m uvicorn api.main:app --reload --port 8000
# Swagger : http://localhost:8000/docs
```

### Lancer avec Docker

```bash
docker-compose up --build
# API    : http://localhost:8000/docs
# Gradio : http://localhost:7860
```

### Tests

```bash
pip install pytest pytest-cov httpx
pytest tests/ -v --cov=api --cov-report=term-missing
# 31 tests, ~95% coverage
```

### Générer des logs et lancer le monitoring

```bash
# Générer 500 logs de production simulés (API doit tourner sur :8000)
python scripts/generate_logs.py --n 500

# Dashboard Streamlit
pip install -r requirements-monitoring.txt
streamlit run monitoring/dashboard.py
# http://localhost:8501
```

### Analyse de drift

Ouvrir `notebooks/07_data_drift.ipynb` — compare la distribution des inputs de production avec les données d'entraînement via Evidently (tests KS et Chi²).

### Déploiement manuel sur HF Spaces

```bash
# Créer un fichier .env avec HF_TOKEN=<ton_token>
python deploy_api.py     # déploie l'API FastAPI
python deploy_gradio.py  # déploie l'interface Gradio
```

### CI/CD

Le workflow `.github/workflows/ci-cd.yml` se déclenche sur chaque push vers `main` :
1. Exécute les 31 tests pytest
2. Si succès → redéploie automatiquement les deux Spaces HF

---

## Stack technique

| Couche | Outils |
|---|---|
| ML | LightGBM, scikit-learn, Optuna, MLflow, SHAP |
| API | FastAPI, Pydantic, uvicorn |
| Interface | Gradio (HF Spaces) |
| Monitoring | Evidently AI (drift), Streamlit (dashboard) |
| Logs | JSONL local (`logs/predictions.jsonl`) |
| Conteneurisation | Docker, docker-compose |
| CI/CD | GitHub Actions → Hugging Face Spaces |

> **Données non versionnées** — télécharger depuis [Kaggle](https://www.kaggle.com/c/home-credit-default-risk/data) et placer dans `data/`.
