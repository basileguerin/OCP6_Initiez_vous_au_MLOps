"""
Tests unitaires de l'API FastAPI — Scoring Crédit.

On couvre :
  - L'endpoint /health
  - L'endpoint /predict : cas nominal, champs manquants, types incorrects,
    valeurs hors plage
"""
import pytest


# ─── Tests /health ────────────────────────────────────────────────────────────

class TestHealth:
    def test_health_retourne_200(self, client):
        """L'API répond 200 quand le modèle est bien chargé."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_contient_status_ok(self, client):
        """Le corps de la réponse confirme que le statut est 'ok'."""
        response = client.get("/health")
        data = response.json()
        assert data["status"] == "ok"
        assert data["model_loaded"] is True

    def test_health_indique_nombre_features(self, client):
        """Le modèle de production utilise exactement 20 features."""
        response = client.get("/health")
        assert response.json()["n_features"] == 20


# ─── Tests /predict — cas nominal ─────────────────────────────────────────────

class TestPredictNominal:
    def test_predict_retourne_200(self, client, payload_valide):
        """Une requête valide doit retourner un statut 200."""
        response = client.post("/predict", json=payload_valide)
        assert response.status_code == 200

    def test_predict_contient_champs_attendus(self, client, payload_valide):
        """La réponse doit contenir score, decision, seuil et latence_ms."""
        data = client.post("/predict", json=payload_valide).json()
        assert "score" in data
        assert "decision" in data
        assert "seuil" in data
        assert "latence_ms" in data

    def test_predict_score_entre_0_et_1(self, client, payload_valide):
        """Le score de défaut est toujours une probabilité entre 0 et 1."""
        score = client.post("/predict", json=payload_valide).json()["score"]
        assert 0.0 <= score <= 1.0

    def test_predict_seuil_est_0_53(self, client, payload_valide):
        """Le seuil de décision métier est figé à 0.53."""
        seuil = client.post("/predict", json=payload_valide).json()["seuil"]
        assert seuil == 0.53

    def test_predict_decision_est_accepte_ou_refuse(self, client, payload_valide):
        """La décision ne peut être que 'ACCEPTE' ou 'REFUSE'."""
        decision = client.post("/predict", json=payload_valide).json()["decision"]
        assert decision in ("ACCEPTE", "REFUSE")

    def test_predict_bon_client_est_accepte(self, client, payload_valide):
        """Un bon client (scores externes élevés) doit être accepté."""
        # EXT_SOURCE élevés → faible risque → ACCEPTE
        response = client.post("/predict", json=payload_valide)
        assert response.json()["decision"] == "ACCEPTE"

    def test_predict_mauvais_client_est_refuse(self, client):
        """Un client à fort risque (scores externes très bas) doit être refusé."""
        payload_risque = {
            "EXT_SOURCE_2": 0.05,
            "EXT_SOURCE_3": 0.04,
            "EXT_SOURCE_1": 0.03,
            "RATIO_CREDIT_BIEN": 1.5,
            "INSTALL_MANQUE_MOYEN": 5000.0,
            "CODE_GENDER_M": 1.0,
            "DAYS_EMPLOYED": -180.0,
            "AMT_ANNUITY": 45000.0,
            "FLAG_OWN_CAR": 0.0,
            "CC_SOLDE_MOYEN": 0.0,
            "BUREAU_JOURS_MOYEN": 0.0,
            "POS_NB_MOIS": 6.0,
            "AGE": 25.0,
            "AMT_CREDIT": 500000.0,
            "BUREAU_DETTE_MOYENNE": 150000.0,
            "NAME_EDUCATION_TYPE_Higher_education": 0.0,
            "PREV_NB_REFUSEES": 5.0,
            "NAME_FAMILY_STATUS_Married": 0.0,
            "DAYS_ID_PUBLISH": -100.0,
            "BUREAU_NB_ACTIFS": 0.0,
        }
        response = client.post("/predict", json=payload_risque)
        assert response.json()["decision"] == "REFUSE"


# ─── Tests /predict — validation des entrées ──────────────────────────────────

class TestPredictValidation:
    def test_champ_manquant_retourne_422(self, client, payload_valide):
        """Un champ obligatoire manquant doit retourner une erreur 422."""
        payload_incomplet = {k: v for k, v in payload_valide.items()
                             if k != "EXT_SOURCE_2"}
        response = client.post("/predict", json=payload_incomplet)
        assert response.status_code == 422

    def test_type_incorrect_retourne_422(self, client, payload_valide):
        """Un type incorrect (texte au lieu de float) doit retourner 422."""
        payload_mauvais_type = {**payload_valide, "AGE": "quarante-cinq"}
        response = client.post("/predict", json=payload_mauvais_type)
        assert response.status_code == 422

    def test_ext_source_superieur_a_1_retourne_422(self, client, payload_valide):
        """EXT_SOURCE doit être entre 0 et 1 — au-dessus retourne 422."""
        payload_hors_plage = {**payload_valide, "EXT_SOURCE_2": 1.5}
        response = client.post("/predict", json=payload_hors_plage)
        assert response.status_code == 422

    def test_ext_source_negatif_retourne_422(self, client, payload_valide):
        """EXT_SOURCE doit être >= 0 — une valeur négative retourne 422."""
        payload_negatif = {**payload_valide, "EXT_SOURCE_3": -0.1}
        response = client.post("/predict", json=payload_negatif)
        assert response.status_code == 422

    def test_age_negatif_retourne_422(self, client, payload_valide):
        """Un âge négatif (-5 ans) n'a pas de sens métier et retourne 422."""
        payload_age_invalide = {**payload_valide, "AGE": -5.0}
        response = client.post("/predict", json=payload_age_invalide)
        assert response.status_code == 422

    def test_age_trop_eleve_retourne_422(self, client, payload_valide):
        """Un âge de 150 ans est biologiquement impossible — retourne 422."""
        payload_age_invalide = {**payload_valide, "AGE": 150.0}
        response = client.post("/predict", json=payload_age_invalide)
        assert response.status_code == 422

    def test_montant_credit_zero_retourne_422(self, client, payload_valide):
        """Un montant de crédit nul n'est pas accepté (doit être > 0)."""
        payload_credit_zero = {**payload_valide, "AMT_CREDIT": 0.0}
        response = client.post("/predict", json=payload_credit_zero)
        assert response.status_code == 422

    def test_montant_credit_negatif_retourne_422(self, client, payload_valide):
        """Un montant de crédit négatif est impossible métier — retourne 422."""
        payload_credit_negatif = {**payload_valide, "AMT_CREDIT": -10000.0}
        response = client.post("/predict", json=payload_credit_negatif)
        assert response.status_code == 422

    def test_nb_refuses_negatif_retourne_422(self, client, payload_valide):
        """Un nombre de demandes refusées négatif est impossible — retourne 422."""
        payload = {**payload_valide, "PREV_NB_REFUSEES": -1.0}
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_body_vide_retourne_422(self, client):
        """Un body vide doit retourner 422 (aucun champ fourni)."""
        response = client.post("/predict", json={})
        assert response.status_code == 422
