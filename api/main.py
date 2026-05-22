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
from fastapi.responses import RedirectResponse

from api.model_loader import load_model
from api.schemas import ClientFeatures, PredictionResponse
import api.model_loader as ml
import api.logger as logger


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

@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")


@app.get("/health", tags=["Monitoring"])
def health():
    """Vérifie que l'API est opérationnelle et que le modèle est chargé."""
    if not ml.is_loaded():
        raise HTTPException(status_code=503, detail="Modèle non chargé")
    return {"status": "ok", "model_loaded": True, "n_features": len(ml.FEATURES),
            "inference_mode": ml.inference_mode()}


@app.post("/predict", response_model=PredictionResponse, tags=["Prédiction"])
def predict(client: ClientFeatures):
    """
    Prédit la probabilité de défaut de paiement pour un client.

    - **score** : probabilité entre 0 et 1 (1 = défaut certain)
    - **decision** : REFUSE si score >= 0.53, ACCEPTE sinon
    - **seuil** : seuil de décision métier (toujours 0.53)
    - **latence_ms** : temps de traitement en millisecondes
    """
    if not ml.is_loaded():
        raise HTTPException(status_code=503, detail="Modèle non chargé")

    debut_total = time.time()

    try:
        valeurs = [getattr(client, f) for f in ml.FEATURES]

        debut_inference = time.time()
        score = ml.predict_score(valeurs)
        inference_ms = round((time.time() - debut_inference) * 1000, 2)

        decision = "REFUSE" if score >= ml.SEUIL else "ACCEPTE"
        response_ms = round((time.time() - debut_total) * 1000, 2)

        logger.log_prediction(
            features=client.model_dump(),
            score=round(score, 4),
            decision=decision,
            inference_ms=inference_ms,
            response_ms=response_ms,
        )

        return PredictionResponse(
            score=round(score, 4),
            decision=decision,
            seuil=ml.SEUIL,
            latence_ms=inference_ms,
        )

    except Exception as e:
        response_ms = round((time.time() - debut_total) * 1000, 2)
        logger.log_erreur(error_message=str(e), response_ms=response_ms)
        raise HTTPException(status_code=500, detail=str(e))
