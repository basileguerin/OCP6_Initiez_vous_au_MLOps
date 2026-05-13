"""
Schémas Pydantic pour la validation des entrées et sorties de l'API.
Chaque champ correspond à une des 20 features sélectionnées par SHAP (notebook 06).
"""
from pydantic import BaseModel, Field
from typing import Annotated


# ─── Entrée : les 20 features dans l'ordre SHAP décroissant ───────────────────

class ClientFeatures(BaseModel):
    """Features d'un client pour la prédiction de défaut crédit.

    Les noms de champs correspondent exactement aux colonnes du modèle.
    Les contraintes métier sont vérifiées avant toute inférence.
    """

    # Scores de solvabilité externes — les plus importants (SHAP #1, #2, #3)
    EXT_SOURCE_2: Annotated[float, Field(ge=0.0, le=1.0,
        description="Score de solvabilité externe bureau 2 (entre 0 et 1)")]
    EXT_SOURCE_3: Annotated[float, Field(ge=0.0, le=1.0,
        description="Score de solvabilité externe bureau 3 (entre 0 et 1)")]
    EXT_SOURCE_1: Annotated[float, Field(ge=0.0, le=1.0,
        description="Score de solvabilité externe bureau 1 (entre 0 et 1)")]

    # Variables financières
    RATIO_CREDIT_BIEN: Annotated[float, Field(gt=0.0,
        description="Ratio montant crédit / valeur du bien (doit être > 0)")]
    INSTALL_MANQUE_MOYEN: Annotated[float, Field(
        description="Manque moyen sur paiements échelonnés (négatif si surpaiement)")]
    AMT_ANNUITY: Annotated[float, Field(gt=0.0,
        description="Montant de l'annuité du crédit (doit être > 0)")]
    AMT_CREDIT: Annotated[float, Field(gt=0.0,
        description="Montant total du crédit demandé (doit être > 0)")]
    CC_SOLDE_MOYEN: Annotated[float, Field(
        description="Solde moyen carte de crédit (négatif si solde créditeur)")]

    # Variables démographiques
    CODE_GENDER_M: Annotated[float, Field(ge=0.0, le=1.0,
        description="Genre masculin : 1 = homme, 0 = femme")]
    FLAG_OWN_CAR: Annotated[float, Field(ge=0.0, le=1.0,
        description="Possède une voiture : 1 = oui, 0 = non")]
    AGE: Annotated[float, Field(gt=18.0, lt=100.0,
        description="Âge du client en années (entre 18 et 100)")]
    NAME_EDUCATION_TYPE_Higher_education: Annotated[float, Field(ge=0.0, le=1.0,
        description="Niveau d'études supérieur : 1 = oui, 0 = non")]
    NAME_FAMILY_STATUS_Married: Annotated[float, Field(ge=0.0, le=1.0,
        description="Statut marital marié : 1 = oui, 0 = non")]

    # Variables emploi (jours négatifs : DAYS_EMPLOYED=-1000 = 1000 jours d'ancienneté)
    DAYS_EMPLOYED: Annotated[float, Field(
        description="Ancienneté dans l'emploi en jours (valeur négative ou 0)")]
    DAYS_ID_PUBLISH: Annotated[float, Field(le=0.0,
        description="Ancienneté du document d'identité en jours (valeur négative)")]

    # Historique bureau de crédit
    BUREAU_JOURS_MOYEN: Annotated[float, Field(
        description="Durée moyenne des crédits au bureau (valeur négative)")]
    BUREAU_DETTE_MOYENNE: Annotated[float, Field(
        description="Dette moyenne enregistrée au bureau (négatif si client créancier net)")]
    BUREAU_NB_ACTIFS: Annotated[float, Field(ge=0.0,
        description="Nombre de crédits actifs au bureau (>= 0)")]

    # Historique POS et demandes précédentes
    POS_NB_MOIS: Annotated[float, Field(ge=0.0,
        description="Nombre de mois dans POS cash balance (>= 0)")]
    PREV_NB_REFUSEES: Annotated[float, Field(ge=0.0,
        description="Nombre de demandes précédentes refusées (>= 0)")]

    model_config = {"json_schema_extra": {
        "example": {
            "EXT_SOURCE_2": 0.62,
            "EXT_SOURCE_3": 0.55,
            "EXT_SOURCE_1": 0.48,
            "RATIO_CREDIT_BIEN": 1.10,
            "INSTALL_MANQUE_MOYEN": 0.0,
            "CODE_GENDER_M": 0.0,
            "DAYS_EMPLOYED": -2500.0,
            "AMT_ANNUITY": 18000.0,
            "FLAG_OWN_CAR": 0.0,
            "CC_SOLDE_MOYEN": 5000.0,
            "BUREAU_JOURS_MOYEN": -800.0,
            "POS_NB_MOIS": 24.0,
            "AGE": 42.0,
            "AMT_CREDIT": 270000.0,
            "BUREAU_DETTE_MOYENNE": 0.0,
            "NAME_EDUCATION_TYPE_Higher_education": 0.0,
            "PREV_NB_REFUSEES": 0.0,
            "NAME_FAMILY_STATUS_Married": 1.0,
            "DAYS_ID_PUBLISH": -1500.0,
            "BUREAU_NB_ACTIFS": 1.0
        }
    }}


# ─── Sortie : résultat de la prédiction ───────────────────────────────────────

class PredictionResponse(BaseModel):
    """Réponse de l'endpoint /predict."""

    score: float = Field(description="Probabilité de défaut (entre 0 et 1)")
    decision: str = Field(description="Décision finale : ACCEPTE ou REFUSE")
    seuil: float = Field(description="Seuil de décision utilisé (0.53)")
    latence_ms: float = Field(description="Temps de traitement en millisecondes")
