---
title: Scoring Credit API
emoji: 🏦
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8000
pinned: false
---

# API Scoring Crédit — Prêt à dépenser

API FastAPI de scoring crédit. Prédit la probabilité de défaut d'un client.

- `GET /health` — statut de l'API
- `POST /predict` — score de défaut (0-1) + décision ACCEPTE/REFUSE
- `GET /docs` — Swagger UI
