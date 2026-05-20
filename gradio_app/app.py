"""Interface Gradio de démo — Scoring Crédit (appel API FastAPI)."""
import os

import gradio as gr
import httpx

# En local : http://localhost:8000
# Sur HF Spaces : variable d'env API_URL configurée dans les secrets du Space
API_URL = os.getenv("API_URL", "http://localhost:8000") + "/predict"


def predire(
    ext_source_2, ext_source_3, ext_source_1,
    ratio_credit_bien, install_manque_moyen, code_gender_m,
    days_employed, amt_annuity, flag_own_car, cc_solde_moyen,
    bureau_jours_moyen, pos_nb_mois, age, amt_credit,
    bureau_dette_moyenne, name_education_higher, prev_nb_refusees,
    name_family_married, days_id_publish, bureau_nb_actifs,
) -> tuple[float, str, str]:
    payload = {
        "EXT_SOURCE_2": ext_source_2,
        "EXT_SOURCE_3": ext_source_3,
        "EXT_SOURCE_1": ext_source_1,
        "RATIO_CREDIT_BIEN": ratio_credit_bien,
        "INSTALL_MANQUE_MOYEN": install_manque_moyen,
        "CODE_GENDER_M": code_gender_m,
        "DAYS_EMPLOYED": days_employed,
        "AMT_ANNUITY": amt_annuity,
        "FLAG_OWN_CAR": flag_own_car,
        "CC_SOLDE_MOYEN": cc_solde_moyen,
        "BUREAU_JOURS_MOYEN": bureau_jours_moyen,
        "POS_NB_MOIS": pos_nb_mois,
        "AGE": age,
        "AMT_CREDIT": amt_credit,
        "BUREAU_DETTE_MOYENNE": bureau_dette_moyenne,
        "NAME_EDUCATION_TYPE_Higher_education": name_education_higher,
        "PREV_NB_REFUSEES": prev_nb_refusees,
        "NAME_FAMILY_STATUS_Married": name_family_married,
        "DAYS_ID_PUBLISH": days_id_publish,
        "BUREAU_NB_ACTIFS": bureau_nb_actifs,
    }

    try:
        resp = httpx.post(API_URL, json=payload, timeout=30.0)
        resp.raise_for_status()
    except httpx.HTTPStatusError as e:
        detail = e.response.json().get("detail", str(e))
        raise gr.Error(f"Erreur API ({e.response.status_code}) : {detail}")
    except httpx.RequestError as e:
        raise gr.Error(f"API injoignable ({API_URL}) : {e}")

    data = resp.json()
    score = data["score"]
    decision = "REFUSÉ ❌" if data["decision"] == "REFUSE" else "ACCEPTÉ ✅"

    if score < 0.3:
        niveau = "🟢 Risque faible"
    elif score < data["seuil"]:
        niveau = "🟡 Risque modéré"
    else:
        niveau = "🔴 Risque élevé"

    return score, decision, niveau


with gr.Blocks(title="Scoring Crédit — Prêt à dépenser", theme=gr.themes.Soft()) as demo:

    gr.Markdown("# 🏦 Scoring Crédit — Prêt à dépenser\nProbabilité de défaut de paiement. Seuil de décision : **0.53**.")

    with gr.Group():
        gr.Markdown("### Scores externes de crédit")
        with gr.Row():
            ext_source_2 = gr.Number(label="EXT_SOURCE_2", value=0.5, minimum=0.0, maximum=1.0)
            ext_source_3 = gr.Number(label="EXT_SOURCE_3", value=0.5, minimum=0.0, maximum=1.0)
            ext_source_1 = gr.Number(label="EXT_SOURCE_1", value=0.5, minimum=0.0, maximum=1.0)

    with gr.Group():
        gr.Markdown("### Crédit demandé")
        with gr.Row():
            ratio_credit_bien = gr.Number(label="RATIO_CREDIT_BIEN", value=0.5, minimum=0.0)
            amt_credit = gr.Number(label="AMT_CREDIT (€)", value=200_000.0, minimum=0.0)
            amt_annuity = gr.Number(label="AMT_ANNUITY (€/an)", value=15_000.0, minimum=0.0)

    with gr.Group():
        gr.Markdown("### Comportement de paiement")
        with gr.Row():
            install_manque_moyen = gr.Number(label="INSTALL_MANQUE_MOYEN", value=0.0)
            cc_solde_moyen = gr.Number(label="CC_SOLDE_MOYEN (€)", value=0.0)
            prev_nb_refusees = gr.Number(label="PREV_NB_REFUSEES", value=0.0, minimum=0.0)

    with gr.Group():
        gr.Markdown("### Historique bureau de crédit")
        with gr.Row():
            bureau_jours_moyen = gr.Number(label="BUREAU_JOURS_MOYEN", value=1000.0)
            bureau_dette_moyenne = gr.Number(label="BUREAU_DETTE_MOYENNE (€)", value=5000.0)
            bureau_nb_actifs = gr.Number(label="BUREAU_NB_ACTIFS", value=1.0, minimum=0.0)

    with gr.Group():
        gr.Markdown("### Informations personnelles")
        with gr.Row():
            age = gr.Number(label="ÂGE (années)", value=35.0, minimum=18.0, maximum=100.0)
            days_employed = gr.Number(label="DAYS_EMPLOYED", value=-2000.0)
            days_id_publish = gr.Number(label="DAYS_ID_PUBLISH", value=-1000.0)
        with gr.Row():
            code_gender_m = gr.Radio(label="Genre", choices=[("Femme", 0), ("Homme", 1)], value=0)
            flag_own_car = gr.Radio(label="Possède une voiture", choices=[("Non", 0), ("Oui", 1)], value=0)
            name_education_higher = gr.Radio(label="Études supérieures", choices=[("Non", 0), ("Oui", 1)], value=0)
            name_family_married = gr.Radio(label="Marié(e)", choices=[("Non", 0), ("Oui", 1)], value=0)

    with gr.Group():
        gr.Markdown("### POS Cash Balance")
        pos_nb_mois = gr.Number(label="POS_NB_MOIS", value=12.0, minimum=0.0)

    btn = gr.Button("🔍 Calculer le score de crédit", variant="primary", size="lg")

    with gr.Row():
        out_score = gr.Number(label="Score de défaut", precision=4)
        out_decision = gr.Textbox(label="Décision")
        out_niveau = gr.Textbox(label="Niveau de risque")

    # inputs → arguments de predire() dans l'ordre ; outputs → valeurs retournées
    btn.click(
        fn=predire,
        inputs=[
            ext_source_2, ext_source_3, ext_source_1,
            ratio_credit_bien, install_manque_moyen, code_gender_m,
            days_employed, amt_annuity, flag_own_car, cc_solde_moyen,
            bureau_jours_moyen, pos_nb_mois, age, amt_credit,
            bureau_dette_moyenne, name_education_higher, prev_nb_refusees,
            name_family_married, days_id_publish, bureau_nb_actifs,
        ],
        outputs=[out_score, out_decision, out_niveau],
    )


if __name__ == "__main__":
    # server_name="0.0.0.0" obligatoire dans Docker pour accepter les connexions externes
    demo.launch(server_name="0.0.0.0", server_port=7860)
