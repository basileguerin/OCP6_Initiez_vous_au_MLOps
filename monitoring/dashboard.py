"""Dashboard Streamlit — Monitoring de l'API Scoring Crédit."""
import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from evidently import Report
from evidently.presets import DataDriftPreset

ROOT      = Path(__file__).parent.parent
LOG_FILE  = ROOT / "logs" / "predictions.jsonl"
DATA_REF  = ROOT / "data" / "processed" / "dataset_final.csv"
SEUIL     = 0.53

FEATURES = [
    "EXT_SOURCE_2", "EXT_SOURCE_3", "EXT_SOURCE_1",
    "RATIO_CREDIT_BIEN", "INSTALL_MANQUE_MOYEN", "CODE_GENDER_M",
    "DAYS_EMPLOYED", "AMT_ANNUITY", "FLAG_OWN_CAR", "CC_SOLDE_MOYEN",
    "BUREAU_JOURS_MOYEN", "POS_NB_MOIS", "AGE", "AMT_CREDIT",
    "BUREAU_DETTE_MOYENNE", "NAME_EDUCATION_TYPE_Higher_education",
    "PREV_NB_REFUSEES", "NAME_FAMILY_STATUS_Married",
    "DAYS_ID_PUBLISH", "BUREAU_NB_ACTIFS",
]

# ---------------------------------------------------------------------------
# Chargement des logs
# ---------------------------------------------------------------------------
@st.cache_data(ttl=30)  # rafraîchissement toutes les 30 secondes
def charger_logs() -> pd.DataFrame:
    if not LOG_FILE.exists() or LOG_FILE.stat().st_size == 0:
        return pd.DataFrame()
    lignes = [json.loads(l) for l in LOG_FILE.read_text().strip().splitlines() if l]
    df = pd.DataFrame([{
        "timestamp":   l["timestamp"],
        "score":       l["score"],
        "decision":    l["decision"],
        "inference_ms": l["inference_ms"],
        "response_ms": l["response_ms"],
        "status":      l["status"],
        **({} if not l["features"] else l["features"]),
    } for l in lignes])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


@st.cache_data(show_spinner="Chargement du dataset de référence…")
def charger_reference(n: int = 500) -> pd.DataFrame:
    col_map = {f: f for f in FEATURES}
    col_map["NAME_EDUCATION_TYPE_Higher_education"] = "NAME_EDUCATION_TYPE_Higher education"
    df = pd.read_csv(DATA_REF, usecols=list(col_map.values())).dropna()
    df = df.rename(columns={v: k for k, v in col_map.items()})
    return df.sample(n=n, random_state=42)[FEATURES].astype(float).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Monitoring — Scoring Crédit", layout="wide", page_icon="🏦")
st.title("🏦 Monitoring — API Scoring Crédit")

df = charger_logs()

if df.empty:
    st.warning("Aucun log trouvé dans `logs/predictions.jsonl`. Lance d'abord `python scripts/generate_logs.py`.")
    st.stop()

df_ok = df[df["status"] == "success"].copy()

# --- KPIs ---
st.markdown("### Métriques globales")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Requêtes totales", len(df))
c2.metric("Taux d'erreur", f"{(df['status']=='error').mean()*100:.1f}%")
c3.metric("Taux de refus", f"{(df_ok['decision']=='REFUSE').mean()*100:.1f}%")
c4.metric("Score moyen", f"{df_ok['score'].mean():.3f}")
c5.metric("Latence p95", f"{df_ok['response_ms'].quantile(0.95):.1f} ms")

st.divider()

# --- Graphiques opérationnels ---
col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Distribution des scores")
    fig = px.histogram(df_ok, x="score", nbins=40,
                       labels={"score": "Score de défaut", "count": "Requêtes"},
                       color_discrete_sequence=["#4C72B0"])
    fig.add_vline(x=SEUIL, line_dash="dash", line_color="red",
                  annotation_text=f"Seuil {SEUIL}", annotation_position="top right")
    fig.update_layout(showlegend=False, height=350)
    st.plotly_chart(fig, width="stretch")

with col2:
    st.markdown("#### Décisions crédit")
    counts = df_ok["decision"].value_counts().reset_index()
    counts.columns = ["decision", "n"]
    fig2 = px.pie(counts, names="decision", values="n",
                  color="decision",
                  color_discrete_map={"ACCEPTE": "#55A868", "REFUSE": "#C44E52"},
                  hole=0.4)
    fig2.update_layout(height=350)
    st.plotly_chart(fig2, width="stretch")

col3, col4 = st.columns(2)

with col3:
    st.markdown("#### Latence d'inférence (ms)")
    fig3 = px.histogram(df_ok, x="inference_ms", nbins=30,
                        labels={"inference_ms": "ms", "count": "Requêtes"},
                        color_discrete_sequence=["#DD8452"])
    fig3.update_layout(height=300)
    st.plotly_chart(fig3, width="stretch")

with col4:
    st.markdown("#### Temps de réponse total (ms)")
    fig4 = px.histogram(df_ok, x="response_ms", nbins=30,
                        labels={"response_ms": "ms", "count": "Requêtes"},
                        color_discrete_sequence=["#8172B2"])
    fig4.update_layout(height=300)
    st.plotly_chart(fig4, width="stretch")

st.divider()

# --- Distribution des features ---
st.markdown("### Distribution des features (prod vs référence)")
st.caption(
    "Distribution d'une feature en production (orange) comparée aux données d'entraînement (bleu). "
    "Un écart visuel important peut signaler un drift."
)

if DATA_REF.exists():
    feat_choisie = st.selectbox("Choisir une feature", FEATURES)
    df_ref_dist  = charger_reference()

    fig_dist = go.Figure()
    fig_dist.add_trace(go.Histogram(
        x=df_ref_dist[feat_choisie],
        name="Référence (train)",
        opacity=0.55,
        marker_color="#4C72B0",
        nbinsx=40,
    ))
    fig_dist.add_trace(go.Histogram(
        x=df_ok[feat_choisie].astype(float),
        name="Production",
        opacity=0.55,
        marker_color="#DD8452",
        nbinsx=40,
    ))
    fig_dist.update_layout(
        barmode="overlay",
        height=380,
        xaxis_title=feat_choisie,
        yaxis_title="Fréquence",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_dist, width="stretch")
else:
    st.info("Dataset de référence non disponible (`data/processed/dataset_final.csv`).")

st.divider()

# --- Table d'analyse statistique des features ---
st.markdown("### Analyse statistique des features clés")
st.caption(
    "Moyennes et écarts-types en référence vs production. "
    "Le **Δ moy.** indique l'écart relatif : 🟢 < 5 %, 🟡 5–15 %, 🔴 > 15 %."
)

if DATA_REF.exists():
    df_ref_stats  = charger_reference()
    df_prod_stats = df_ok[FEATURES].astype(float)

    lignes = []
    for feat in FEATURES:
        mean_ref  = df_ref_stats[feat].mean()
        mean_prod = df_prod_stats[feat].mean()
        delta_pct = ((mean_prod - mean_ref) / abs(mean_ref) * 100) if mean_ref != 0 else 0.0
        lignes.append({
            "Feature":    feat,
            "Moy. réf.":  round(mean_ref, 4),
            "Moy. prod.": round(mean_prod, 4),
            "Δ moy. (%)": round(delta_pct, 1),
            "Std réf.":   round(df_ref_stats[feat].std(), 4),
            "Std prod.":  round(df_prod_stats[feat].std(), 4),
        })

    df_stats = pd.DataFrame(lignes)

    def colorier_delta(val: float) -> str:
        """Colorie la cellule selon l'amplitude du drift."""
        a = abs(val)
        if a < 5:
            return "color: #2a9d2a"
        elif a < 15:
            return "color: #c97a00"
        return "color: #C44E52"

    styled = (
        df_stats.style
        .map(colorier_delta, subset=["Δ moy. (%)"])
        .format({
            "Moy. réf.":  "{:.4f}",
            "Moy. prod.": "{:.4f}",
            "Δ moy. (%)": "{:+.1f}%",
            "Std réf.":   "{:.4f}",
            "Std prod.":  "{:.4f}",
        })
    )
    st.dataframe(styled, width="stretch", hide_index=True)
else:
    st.info("Dataset de référence non disponible (`data/processed/dataset_final.csv`).")

st.divider()

# --- Analyse de data drift ---
st.markdown("### Analyse de data drift (Evidently)")
st.caption(
    "Comparaison entre la distribution d'entraînement (référence) et les inputs reçus en production. "
    "Test KS pour les variables continues, Chi² pour les binaires."
)

features_prod = [f for f in FEATURES if f in df_ok.columns]
if len(features_prod) == len(FEATURES):
    if st.button("▶ Lancer l'analyse de drift", type="primary"):
        with st.spinner("Analyse en cours…"):
            df_ref  = charger_reference(n=min(len(df_ok), 500))
            df_prod = df_ok[FEATURES].astype(float).reset_index(drop=True)

            report   = Report(metrics=[DataDriftPreset()])
            snapshot = report.run(reference_data=df_ref, current_data=df_prod)
            # as_iframe=False : HTML brut sans wrapper iframe interne
            # Streamlit gère seul la hauteur via st.components.v1.html
            html = snapshot.get_html_str(as_iframe=False)

        st.components.v1.html(html, height=4000, scrolling=True)
else:
    st.info("Les features ne sont pas toutes présentes dans les logs — drift non disponible.")
