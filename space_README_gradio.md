---
title: Scoring Credit Demo
emoji: 🏦
colorFrom: blue
colorTo: indigo
sdk: gradio
app_file: app.py
app_port: 7860
pinned: false
---

# Démo Scoring Crédit — Prêt à dépenser

Interface de démonstration du modèle de scoring crédit.
Renseignez les 20 features du client pour obtenir la probabilité de défaut et la décision crédit.

- Seuil de décision : **0.53** (optimisé sur coût métier FN = 10× FP)
- Modèle : LightGBM, 20 features sélectionnées par SHAP, AUC = 0.768
- API : [scoring-credit-api](https://huggingface.co/spaces/basmoket/scoring-credit-api)
