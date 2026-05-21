## 📌 Project Overview

End-to-end fraud detection system combining:
- **Machine Learning** (LightGBM, XGBoost, Isolation Forest)
- **Class Imbalance Handling** (SMOTE)
- **Explainable AI** (SHAP values)
- **Interactive Dashboard** (Streamlit + Plotly)

---

## 📁 Project Structure

```
FraudDetection_Abhaykumar
├── analysis.ipynb              ← Main Jupyter Notebook (all 8 tasks)
├── dashboard/
│   ├── app.py                  ← Streamlit dashboard (3-page app)
│   ├── model.pkl               ← Trained LightGBM + XGBoost models
│   └── risk_results.csv        ← Test set predictions + risk tier
├── data/
│   ├── train_transaction.csv   ← (download from Kaggle)
│   └── train_identity.csv      ← (download from Kaggle)
├── charts/
│   ├── class_imbalance.png
│   ├── txn_amt_dist.png
│   ├── corr_heatmap.png
│   ├── fraud_by_hour.png
│   ├── roc_curve.png
│   ├── pr_curve.png
│   ├── confusion_lightgbm.png
│   ├── confusion_xgboost.png
│   ├── threshold_f1.png
│   ├── risk_tier_donut.png
│   ├── shap_waterfall_fraud.png
│   ├── shap_waterfall_borderline.png
│   ├── shap_waterfall_legit.png
│   └── shap_dependence.png
├── model_comparison.png
├── shap_summary.png
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup & Installation

```bash
# 1. Clone / unzip the project
cd FraudDetection

# 2. Install dependencies
pip install -r requirements.txt

# 3. Place dataset files in data/
#    train_transaction.csv
#    train_identity.csv

# 4. Run the Jupyter Notebook
jupyter notebook analysis.ipynb

# 5. Launch the dashboard
cd dashboard
streamlit run app.py
```

---

## 🧠 Model Results

| Model      | ROC-AUC | PR-AUC | F1 Score |
|------------|---------|--------|----------|
| LightGBM   | 0.9399  | 0.6463 | 0.6146   |
| XGBoost    | 0.9000  | 0.5210 | ~0.57    |

**Best Model: LightGBM** — highest ROC-AUC and PR-AUC on imbalanced data.

---

## 🚦 Risk Tiers

| Tier        | Probability     | Action              |
|-------------|-----------------|---------------------|
| 🔴 Critical | ≥ 0.75          | Block immediately   |
| 🟡 Suspicious | 0.40 – 0.74   | Step-up auth        |
| 🟢 Clear    | < 0.40          | Auto-approve        |

---

## 📊 Dashboard Pages

1. **Overview** — KPIs, tier distribution, fraud by hour
2. **Transaction Explorer** — Search by ID, filter, probability scatter
3. **SHAP Explainer** — Per-transaction explanation in plain English

### Live URL
> Deploy to [Streamlit Community Cloud](https://streamlit.io/cloud):  
> `streamlit run dashboard/app.py`  
> Add live URL here after deployment.
http://localhost:8501
---

## 🔑 Key Findings

- **Top fraud signals (SHAP):** V258, V257, card1, TransactionAmt, C14
- **Critical risk transactions** tend to have unusual device fingerprints and high C-feature values
- **PR-AUC matters more than accuracy** because the dataset is 96.5% legitimate — a trivial model gets 96.5% accuracy by predicting everything as legit
- **Estimated savings:** At a 3.5% fraud rate on $5T global losses, catching 53% of fraud = ~$92.75B saved annually

---
