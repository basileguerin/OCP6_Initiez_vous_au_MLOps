"""Logging des prédictions vers un fichier JSONL (une ligne JSON par requête)."""
import json
import threading
from datetime import datetime, timezone
from pathlib import Path

_LOG_FILE = Path(__file__).parent.parent / "logs" / "predictions.jsonl"

# Verrou pour éviter les écritures concurrentes (FastAPI est asynchrone)
_lock = threading.Lock()

MODEL_VERSION = "v2-lgbm-20features"


def log_prediction(features: dict, score: float, decision: str,
                   inference_ms: float, response_ms: float) -> None:
    _ecrire({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "features": features,
        "score": score,
        "decision": decision,
        "inference_ms": inference_ms,
        "response_ms": response_ms,
        "status": "success",
        "model_version": MODEL_VERSION,
        "error_message": None,
    })


def log_erreur(error_message: str, response_ms: float) -> None:
    _ecrire({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "features": None,
        "score": None,
        "decision": None,
        "inference_ms": None,
        "response_ms": response_ms,
        "status": "error",
        "model_version": MODEL_VERSION,
        "error_message": error_message,
    })


def _ecrire(entry: dict) -> None:
    with _lock:
        _LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(_LOG_FILE, "a") as f:
            f.write(json.dumps(entry) + "\n")
