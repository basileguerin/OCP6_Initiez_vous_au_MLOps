"""
API FastAPI — Scoring Crédit «Prêt à dépenser»

Deux endpoints :
  GET  /health   → vérifie que l'API est opérationnelle
  POST /predict  → retourne le score de défaut et la décision (ACCEPTE / REFUSE)

Le modèle est chargé une seule fois au démarrage via le mécanisme `lifespan`,
jamais à chaque requête.
"""
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from api.model_loader import load_model, FEATURES, SEUIL
from api.schemas import ClientFeatures, PredictionResponse
import api.model_loader as ml


# ─── Chargement du modèle au démarrage ───────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    lifespan est le gestionnaire de cycle de vie de FastAPI.
    Le code avant 'yield' s'exécute au démarrage, après 'yield' à l'arrêt.
    C'est ici qu'on charge le modèle — une seule fois.
    """
    load_model()
    yield
    # Nettoyage éventuel à l'arrêt (ici rien à faire)


# ─── Application FastAPI ──────────────────────────────────────────────────────

app = FastAPI(
    title="API Scoring Crédit — Prêt à dépenser",
    description=(
        "Prédit la probabilité de défaut d'un client à partir de ses 20 features "
        "les plus importantes (sélectionnées par SHAP). "
        "Seuil de décision métier : **0.53** (FN coûte 10× plus qu'un FP)."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health", tags=["Monitoring"])
def health():
    """Vérifie que l'API est opérationnelle et que le modèle est chargé."""
    if ml.model is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")
    return {"status": "ok", "model_loaded": True, "n_features": len(ml.FEATURES)}


@app.post("/predict", response_model=PredictionResponse, tags=["Prédiction"])
def predict(client: ClientFeatures):
    """
    Prédit la probabilité de défaut de paiement pour un client.

    - **score** : probabilité entre 0 et 1 (1 = défaut certain)
    - **decision** : REFUSE si score >= 0.53, ACCEPTE sinon
    - **seuil** : seuil de décision métier (toujours 0.53)
    - **latence_ms** : temps de traitement en millisecondes
    """
    if ml.model is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")

    debut = time.time()

    # Construction du vecteur dans l'ordre exact des features du modèle
    # getattr(client, feature) récupère la valeur Pydantic de chaque champ
    vecteur = [[getattr(client, feature) for feature in ml.FEATURES]]

    # predict_proba retourne [[proba_classe_0, proba_classe_1]]
    # On garde la colonne 1 : probabilité d'être en défaut
    score = float(ml.model.predict_proba(vecteur)[0][1])

    decision = "REFUSE" if score >= ml.SEUIL else "ACCEPTE"
    latence_ms = round((time.time() - debut) * 1000, 2)

    return PredictionResponse(
        score=round(score, 4),
        decision=decision,
        seuil=ml.SEUIL,
        latence_ms=latence_ms,
    )
