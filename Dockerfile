FROM python:3.12-slim

WORKDIR /app

# libgomp est la librairie OpenMP requise par LightGBM
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

# Dépendances d'abord — couche cachée si le code change mais pas requirements
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# Artefacts du modèle et code de l'API
COPY models/ ./models/
COPY api/ ./api/

ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
