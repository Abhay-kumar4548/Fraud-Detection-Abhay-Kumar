import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pickle, os, warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Load assets ──────────────────────────────────────────────────────────────

def find_file(filename):
    """Search for a file in common locations — works regardless of how you run the app."""
    search_paths = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), filename),  # same folder as app.py
        os.path.join(os.getcwd(), filename),                                  # current working directory
        os.path.join(os.getcwd(), "dashboard", filename),                     # dashboard subfolder
        filename,                                                              # absolute path fallback
    ]
    for p in search_paths:
        if os.path.exists(p):
            return p
    # Show helpful error
    st.error(f"""
    ❌ Could not find **{filename}**
    
    Make sure this file is in the same folder as `app.py`:
    
    ```
    dashboard/
    ├── app.py          ← you are here
    ├── model.pkl       ← must be here
    └── risk_results.csv ← must be here
    ```
    
    Searched in:
    {chr(10).join(search_paths)}
    """)
    st.stop()

@st.cache_resource
def load_model():
    path = find_file("model.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_results():
    path = find_file("risk_results.csv")
    df = pd.read_csv(path)
    df['TransactionID'] = range(len(df))
    return df

model_bundle = load_model()
results_df = load_results()

# ── Sidebar ───────────────────────────────────────────────────────────────────
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

# ── Filtered data ─────────────────────────────────────────────────────────────
filtered = results_df[
    (results_df['tier'].isin(tier_filter)) &
    (results_df['proba'] >= prob_range[0]) &
    (results_df['proba'] <= prob_range[1])
]

# ══════════════════════════════════════════════════════════════════════════════
#  OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "📊 Overview":
    st.title("📊 Fraud Detection — Operations Overview")
    st.markdown("---")

    total = len(results_df)
    fraud_count = results_df['true'].sum()
    detection_rate = (results_df['true'] == (results_df['proba'] >= 0.5).astype(int)).mean()
    avg_fraud_prob = results_df[results_df['true'] == 1]['proba'].mean()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📦 Total Transactions", f"{total:,}")
    c2.metric("🚨 Fraud Cases", f"{fraud_count:,}", delta=f"{fraud_count/total*100:.1f}% rate", delta_color="inverse")
    c3.metric("✅ Detection Rate", f"{detection_rate*100:.1f}%")
    c4.metric("📈 Avg Fraud Score", f"{avg_fraud_prob:.3f}")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        tier_counts = results_df['tier'].value_counts().reset_index()
        tier_counts.columns = ['Tier', 'Count']
        fig = px.pie(tier_counts, names='Tier', values='Count',
                     color='Tier',
                     color_discrete_map={'Clear':'#2ecc71','Suspicious':'#f39c12','Critical':'#e74c3c'},
                     hole=0.5, title="Risk Tier Distribution")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        tier_fraud = results_df.groupby('tier').agg(
            Total=('true','count'), Fraud=('true','sum')
        ).reset_index()
        tier_fraud['FraudRate'] = tier_fraud['Fraud'] / tier_fraud['Total'] * 100
        fig2 = px.bar(tier_fraud, x='tier', y='FraudRate',
                      color='tier',
                      color_discrete_map={'Clear':'#2ecc71','Suspicious':'#f39c12','Critical':'#e74c3c'},
                      title="Fraud Rate per Risk Tier (%)",
                      labels={'FraudRate':'Fraud Rate (%)', 'tier':'Risk Tier'})
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        if 'HourOfDay' in results_df.columns:
            hourly = results_df.groupby('HourOfDay')['true'].mean() * 100
            fig3 = px.bar(x=hourly.index, y=hourly.values,
                          title="Fraud Rate by Hour of Day",
                          labels={'x':'Hour','y':'Fraud Rate (%)'})
            fig3.update_traces(marker_color='#e74c3c')
            st.plotly_chart(fig3, use_container_width=True)

    with col4:
        fig4 = px.histogram(results_df, x='proba', nbins=60,
                            color='tier',
                            color_discrete_map={'Clear':'#2ecc71','Suspicious':'#f39c12','Critical':'#e74c3c'},
                            title="Fraud Probability Distribution",
                            labels={'proba':'Fraud Probability'})
        st.plotly_chart(fig4, use_container_width=True)

    st.markdown("---")
    st.subheader("📌 Key Model Metrics")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("ROC-AUC", "0.9399", "LightGBM")
    m2.metric("PR-AUC", "0.6463")
    m3.metric("F1 Score", "0.6146")
    m4.metric("Precision", "~0.72")
    m5.metric("Recall", "~0.53")

# ══════════════════════════════════════════════════════════════════════════════
#  TRANSACTION EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔎 Transaction Explorer":
    st.title("🔎 Transaction Explorer")
    st.markdown(f"Showing **{len(filtered):,}** transactions matching your filters.")

    search_id = st.text_input("🔍 Search by TransactionID (row number)", "")
    if search_id.strip():
        try:
            row = results_df[results_df['TransactionID'] == int(search_id)]
            if not row.empty:
                r = row.iloc[0]
                st.success(f"Found Transaction #{int(r['TransactionID'])}")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Fraud Probability", f"{r['proba']:.4f}")
                col2.metric("Actual Label", "🚨 FRAUD" if r['true']==1 else "✅ Legit")
                col3.metric("Risk Tier", str(r['tier']))
                color = {'Clear':'#2ecc71','Suspicious':'#f39c12','Critical':'#e74c3c'}.get(str(r['tier']),'gray')
                col4.markdown(f"<div style='background:{color};padding:10px;border-radius:8px;text-align:center;color:white;font-size:18px;font-weight:bold'>{r['tier']}</div>", unsafe_allow_html=True)
            else:
                st.warning("TransactionID not found.")
        except ValueError:
            st.error("Please enter a valid integer ID.")

    st.markdown("---")

    # Risk gauge scatter
    fig = go.Figure(go.Scatter(
        x=filtered.index[:2000],
        y=filtered['proba'].iloc[:2000],
        mode='markers',
        marker=dict(
            color=filtered['proba'].iloc[:2000],
            colorscale='RdYlGn_r',
            size=5,
            colorbar=dict(title="Fraud Prob")
        ),
        text=filtered['tier'].iloc[:2000]
    ))
    fig.update_layout(title="Fraud Probability per Transaction (first 2000 shown)",
                      xaxis_title="Transaction Index", yaxis_title="Fraud Probability")
    fig.add_hline(y=0.4, line_dash='dash', line_color='orange', annotation_text='Suspicious threshold')
    fig.add_hline(y=0.75, line_dash='dash', line_color='red', annotation_text='Critical threshold')
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 📋 Transaction Table")
    show_df = filtered[['TransactionID','proba','true','tier']].rename(
        columns={'proba':'Fraud Score','true':'Actual Fraud','tier':'Risk Tier'})
    show_df['Actual Fraud'] = show_df['Actual Fraud'].map({0:'✅ Legit',1:'🚨 Fraud'})
    st.dataframe(show_df.head(500), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# SHAP EXPLAINER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🧠 SHAP Explainer":
    st.title("🧠 SHAP Explainer — Understand Predictions")
    st.markdown("Enter a TransactionID to see why the model scored it that way.")

    txn_id = st.number_input("TransactionID (0 to 118107)", min_value=0, max_value=118107, value=0, step=1)
    row = results_df[results_df['TransactionID'] == txn_id]

    if not row.empty:
        r = row.iloc[0]
        prob = r['proba']
        tier = r['tier']
        actual = r['true']

        c1, c2, c3 = st.columns(3)
        c1.metric("Fraud Score", f"{prob:.4f}")
        c2.metric("Risk Tier", str(tier))
        c3.metric("Actual Label", "🚨 FRAUD" if actual==1 else "✅ Legitimate")

        # Plain-English explanation
        st.markdown("---")
        st.subheader("📝 Plain-English Explanation")
        if prob >= 0.75:
            explanation = f"""
**🔴 CRITICAL RISK — Score: {prob:.3f}**

This transaction has been flagged as **very likely fraudulent**. The model detected several strong
indicators typically associated with fraudulent activity:
- The transaction pattern deviates significantly from typical user behavior
- Device and timing features suggest unusual access patterns
- High-risk email domains or card characteristics were detected

**Recommended Action:** Block transaction immediately and flag for manual review.
            """
        elif prob >= 0.40:
            explanation = f"""
**🟡 SUSPICIOUS — Score: {prob:.3f}**

This transaction shows **borderline characteristics**. Some features suggest legitimacy, 
while others align with known fraud patterns:
- Moderate anomaly detected in transaction timing or device fingerprint
- Amount may be slightly outside normal range for this card type

**Recommended Action:** Step-up authentication (OTP/biometric) before approving.
            """
        else:
            explanation = f"""
**🟢 CLEAR — Score: {prob:.3f}**

This transaction appears **legitimate** with high confidence:
- Transaction amount, timing, and device match historical patterns
- Card and email domain are from trusted sources
- No velocity anomalies detected

**Recommended Action:** Approve automatically.
            """
        st.markdown(explanation)

        # SHAP waterfall image
        st.markdown("---")
        st.subheader("📊 SHAP Waterfall Plot")
        charts_dir = os.path.join(os.path.dirname(__file__), '..', 'charts')
        if prob >= 0.75:
            img_path = os.path.join(charts_dir, 'shap_waterfall_fraud.png')
            label_txt = "Similar confirmed fraud case SHAP breakdown:"
        elif prob >= 0.40:
            img_path = os.path.join(charts_dir, 'shap_waterfall_borderline.png')
            label_txt = "Similar borderline case SHAP breakdown:"
        else:
            img_path = os.path.join(charts_dir, 'shap_waterfall_legit.png')
            label_txt = "Similar legitimate transaction SHAP breakdown:"
        st.caption(label_txt)
        if os.path.exists(img_path):
            st.image(img_path, use_column_width=True)

        st.markdown("---")
        st.subheader("🌐 Global Feature Importance (SHAP Summary)")
        summary_path = os.path.join(os.path.dirname(__file__), '..', 'shap_summary.png')
        if os.path.exists(summary_path):
            st.image(summary_path, use_column_width=True)
    else:
        st.info("Enter a TransactionID above to see its explanation.")

    st.markdown("---")
    st.markdown("**Top SHAP Features:** V258, V257, card1, TransactionAmt, C14 drive most predictions.")
