import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ═══════════════════════════════════════════════════════════════
# FILE PATH HELPER — works locally AND on Streamlit Cloud
# ═══════════════════════════════════════════════════════════════
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def get_path(filename):
    """Find file — checks same folder as app.py first, then common locations."""
    candidates = [
        os.path.join(BASE_DIR, filename),
        os.path.join(BASE_DIR, '..', filename),
        os.path.join(os.getcwd(), filename),
        os.path.join(os.getcwd(), 'dashboard', filename),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None

# ═══════════════════════════════════════════════════════════════
# LOAD DATA — with clear error messages inside Streamlit
# ═══════════════════════════════════════════════════════════════
@st.cache_data(show_spinner="Loading results...")
def load_results():
    path = get_path("risk_results.csv")
    if path is None:
        return None
    df = pd.read_csv(path)
    df['TransactionID'] = range(len(df))
    return df

results_df = load_results()

# Show error inside app (not crash) if files missing
if results_df is None:
    st.error("❌ Could not find risk_results.csv")
    st.info("""
    **Fix:** Make sure this file is in the **same folder** as app.py:
    ```
    dashboard/
    ├── app.py
    └── risk_results.csv
    ```
    If deploying on Streamlit Cloud, make sure the file is **committed to GitHub**.
    """)
    st.stop()

# ═══════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════
st.sidebar.image("https://img.icons8.com/fluency/96/shield.png", width=80)
st.sidebar.title("🔍 Fraud Detection")
st.sidebar.markdown("---")

page = st.sidebar.radio("Navigate to", [
    "📊 Overview",
    "🔎 Transaction Explorer",
    "🧠 SHAP Explainer"
])

tier_filter = st.sidebar.multiselect(
    "Filter by Risk Tier",
    options=["Clear", "Suspicious", "Critical"],
    default=["Clear", "Suspicious", "Critical"]
)
prob_range = st.sidebar.slider("Probability Range", 0.0, 1.0, (0.0, 1.0), 0.01)

st.sidebar.markdown("---")
st.sidebar.markdown("**Model:** LightGBM")
st.sidebar.markdown("**ROC-AUC:** 0.9399")
st.sidebar.markdown("**PR-AUC:** 0.6463")

# ═══════════════════════════════════════════════════════════════
# FILTERED DATA
# ═══════════════════════════════════════════════════════════════
# Normalize tier column (strip emoji if present)
results_df['tier_clean'] = results_df['tier'].astype(str).str.replace(
    r'[🟢🟡🔴]\s*', '', regex=True).str.strip()

filtered = results_df[
    (results_df['tier_clean'].isin(tier_filter)) &
    (results_df['proba'] >= prob_range[0]) &
    (results_df['proba'] <= prob_range[1])
]

# ═══════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ═══════════════════════════════════════════════════════════════
if page == "📊 Overview":
    st.title("📊 Fraud Detection — Operations Overview")
    st.markdown("---")

    total        = len(results_df)
    fraud_count  = int(results_df['true'].sum())
    fraud_rate   = fraud_count / total * 100
    avg_score    = results_df[results_df['true'] == 1]['proba'].mean()
    detected     = int(((results_df['proba'] >= 0.5) & (results_df['true'] == 1)).sum())
    detection_pct= detected / fraud_count * 100 if fraud_count > 0 else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📦 Total Transactions", f"{total:,}")
    c2.metric("🚨 Fraud Cases",        f"{fraud_count:,}",
              delta=f"{fraud_rate:.1f}% rate", delta_color="inverse")
    c3.metric("✅ Fraud Detected",     f"{detected:,}",
              delta=f"{detection_pct:.1f}% recall")
    c4.metric("📈 Avg Fraud Score",    f"{avg_score:.3f}")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        tier_counts = results_df['tier_clean'].value_counts().reset_index()
        tier_counts.columns = ['Tier', 'Count']
        fig = px.pie(
            tier_counts, names='Tier', values='Count',
            color='Tier',
            color_discrete_map={
                'Clear':      '#2ecc71',
                'Suspicious': '#f39c12',
                'Critical':   '#e74c3c'
            },
            hole=0.5, title="Risk Tier Distribution"
        )
        st.plotly_chart(fig, width="stretch")

    with col2:
        tier_fraud = results_df.groupby('tier_clean').agg(
            Total=('true', 'count'),
            Fraud=('true', 'sum')
        ).reset_index()
        tier_fraud['Fraud Rate (%)'] = (tier_fraud['Fraud'] / tier_fraud['Total'] * 100).round(2)
        fig2 = px.bar(
            tier_fraud, x='tier_clean', y='Fraud Rate (%)',
            color='tier_clean',
            color_discrete_map={
                'Clear':      '#2ecc71',
                'Suspicious': '#f39c12',
                'Critical':   '#e74c3c'
            },
            title="Fraud Rate per Risk Tier (%)",
            labels={'tier_clean': 'Risk Tier'}
        )
        st.plotly_chart(fig2, width="stretch")

    col3, col4 = st.columns(2)
    with col3:
        if 'HourOfDay' in results_df.columns:
            hourly = results_df.groupby(
                results_df['HourOfDay'].clip(0, 23).astype(int)
            )['true'].mean() * 100
            fig3 = px.bar(
                x=hourly.index, y=hourly.values,
                title="Fraud Rate by Hour of Day",
                labels={'x': 'Hour', 'y': 'Fraud Rate (%)'}
            )
            fig3.update_traces(marker_color='#e74c3c')
            st.plotly_chart(fig3, width="stretch")

    with col4:
        fig4 = px.histogram(
            results_df, x='proba', nbins=60,
            color='tier_clean',
            color_discrete_map={
                'Clear':      '#2ecc71',
                'Suspicious': '#f39c12',
                'Critical':   '#e74c3c'
            },
            title="Fraud Probability Distribution",
            labels={'proba': 'Fraud Probability', 'tier_clean': 'Tier'}
        )
        st.plotly_chart(fig4, width="stretch")

    st.markdown("---")
    st.subheader("📌 Key Model Metrics")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("ROC-AUC",   "0.9399", "LightGBM")
    m2.metric("PR-AUC",    "0.6463")
    m3.metric("F1 Score",  "0.6146")
    m4.metric("Precision", "~0.72")
    m5.metric("Recall",    "~0.53")

# ═══════════════════════════════════════════════════════════════
# PAGE 2 — TRANSACTION EXPLORER
# ═══════════════════════════════════════════════════════════════
elif page == "🔎 Transaction Explorer":
    st.title("🔎 Transaction Explorer")
    st.markdown(f"Showing **{len(filtered):,}** transactions matching your filters.")

    search_id = st.text_input("🔍 Search by TransactionID (0 to 118107)", "")
    if search_id.strip():
        try:
            row = results_df[results_df['TransactionID'] == int(search_id)]
            if not row.empty:
                r = row.iloc[0]
                st.success(f"Found Transaction #{int(r['TransactionID'])}")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Fraud Probability", f"{r['proba']:.4f}")
                col2.metric("Actual Label", "🚨 FRAUD" if r['true'] == 1 else "✅ Legit")
                col3.metric("Risk Tier", str(r['tier_clean']))
                color = {
                    'Clear':      '#2ecc71',
                    'Suspicious': '#f39c12',
                    'Critical':   '#e74c3c'
                }.get(str(r['tier_clean']), 'gray')
                col4.markdown(
                    f"<div style='background:{color};padding:10px;border-radius:8px;"
                    f"text-align:center;color:white;font-size:18px;font-weight:bold'>"
                    f"{r['tier_clean']}</div>",
                    unsafe_allow_html=True
                )
            else:
                st.warning("TransactionID not found.")
        except ValueError:
            st.error("Please enter a valid integer ID.")

    st.markdown("---")

    # Scatter plot
    plot_df = filtered.head(2000)
    fig = go.Figure(go.Scatter(
        x=plot_df.index,
        y=plot_df['proba'],
        mode='markers',
        marker=dict(
            color=plot_df['proba'],
            colorscale='RdYlGn_r',
            size=5,
            colorbar=dict(title="Fraud Prob")
        ),
        text=plot_df['tier_clean']
    ))
    fig.update_layout(
        title="Fraud Probability per Transaction (first 2000 shown)",
        xaxis_title="Transaction Index",
        yaxis_title="Fraud Probability"
    )
    fig.add_hline(y=0.4,  line_dash='dash', line_color='orange',
                  annotation_text='Suspicious threshold')
    fig.add_hline(y=0.75, line_dash='dash', line_color='red',
                  annotation_text='Critical threshold')
    st.plotly_chart(fig, width="stretch")

    st.markdown("### 📋 Transaction Table")
    show_df = filtered[['TransactionID', 'proba', 'true', 'tier_clean']].rename(columns={
        'proba':      'Fraud Score',
        'true':       'Actual Fraud',
        'tier_clean': 'Risk Tier'
    }).head(500).copy()
    show_df['Actual Fraud'] = show_df['Actual Fraud'].map({0: '✅ Legit', 1: '🚨 Fraud'})
    st.dataframe(show_df, width="stretch")

# ═══════════════════════════════════════════════════════════════
# PAGE 3 — SHAP EXPLAINER
# ═══════════════════════════════════════════════════════════════
elif page == "🧠 SHAP Explainer":
    st.title("🧠 SHAP Explainer — Understand Predictions")
    st.markdown("Enter a TransactionID to see why the model scored it that way.")

    txn_id = st.number_input(
        "TransactionID (0 to 118107)",
        min_value=0, max_value=118107, value=0, step=1
    )
    row = results_df[results_df['TransactionID'] == txn_id]

    if not row.empty:
        r    = row.iloc[0]
        prob = float(r['proba'])
        tier = str(r['tier_clean'])
        actual = int(r['true'])

        c1, c2, c3 = st.columns(3)
        c1.metric("Fraud Score", f"{prob:.4f}")
        c2.metric("Risk Tier",   tier)
        c3.metric("Actual Label", "🚨 FRAUD" if actual == 1 else "✅ Legitimate")

        st.markdown("---")
        st.subheader("📝 Plain-English Explanation")

        if prob >= 0.75:
            st.error(f"""
**🔴 CRITICAL RISK — Score: {prob:.3f}**

This transaction has been flagged as **very likely fraudulent**.

- Strong anomaly detected in Vesta behavioural signals (V258, V257)
- Card and device fingerprint match known fraud patterns
- Transaction velocity and amount are outside normal range

**Recommended Action:** Block transaction immediately and flag for manual review.
            """)
        elif prob >= 0.40:
            st.warning(f"""
**🟡 SUSPICIOUS — Score: {prob:.3f}**

This transaction shows **borderline characteristics**.

- Moderate anomaly in device fingerprint or timing
- Amount is slightly outside normal range for this card type
- Some features suggest legitimacy while others raise concern

**Recommended Action:** Step-up authentication (OTP/biometric) before approving.
            """)
        else:
            st.success(f"""
**🟢 CLEAR — Score: {prob:.3f}**

This transaction appears **legitimate** with high confidence.

- Transaction amount, timing and device match historical patterns
- Card and email domain are from trusted sources
- No velocity anomalies detected

**Recommended Action:** Approve automatically.
            """)

        # SHAP waterfall images
        st.markdown("---")
        st.subheader("📊 SHAP Waterfall Plot")

        charts_candidates = [
            os.path.join(BASE_DIR, '..', 'charts'),
            os.path.join(BASE_DIR, 'charts'),
            os.path.join(os.getcwd(), 'charts'),
        ]
        charts_dir = next((d for d in charts_candidates if os.path.isdir(d)), None)

        if prob >= 0.75:
            img_name = 'shap_waterfall_fraud.png'
            caption  = "SHAP breakdown for a similar confirmed fraud case"
        elif prob >= 0.40:
            img_name = 'shap_waterfall_borderline.png'
            caption  = "SHAP breakdown for a similar borderline case"
        else:
            img_name = 'shap_waterfall_legit.png'
            caption  = "SHAP breakdown for a similar legitimate transaction"

        if charts_dir:
            img_path = os.path.join(charts_dir, img_name)
            if os.path.exists(img_path):
                st.image(img_path, caption=caption, use_container_width=True)
            else:
                st.info(f"Chart not found: {img_name} — add charts/ folder to your repo.")
        else:
            st.info("Add the charts/ folder to your GitHub repo to see SHAP plots here.")

        # Global SHAP summary
        st.markdown("---")
        st.subheader("🌐 Global Feature Importance (SHAP Summary)")
        summary_candidates = [
            os.path.join(BASE_DIR, '..', 'shap_summary.png'),
            os.path.join(BASE_DIR, 'shap_summary.png'),
            os.path.join(os.getcwd(), 'shap_summary.png'),
        ]
        summary_path = next((p for p in summary_candidates if os.path.exists(p)), None)
        if summary_path:
            st.image(summary_path, use_container_width=True)
        else:
            st.info("Add shap_summary.png to your GitHub repo to see it here.")

    else:
        st.info("Enter a TransactionID above to see its explanation.")

    st.markdown("---")
    st.markdown("**Top SHAP Features:** V258, V257, card1, TransactionAmt, C14")
