# 🛡️ FraudGuard: End-to-End Real-Time Credit Card Fraud Detection System

![Python](https://img.shields.io/badge/Python-3.11%2B-blue)
![LightGBM](https://img.shields.io/badge/LightGBM-Latest-green)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-red)
![SHAP](https://img.shields.io/badge/SHAP-Explainable%20AI-orange)
![License](https://img.shields.io/badge/License-MIT-yellow)

**A complete production-ready machine learning system for detecting fraudulent credit card transactions in real time.**

Built as a full walkthrough from raw data → trained model → real-time API → monitoring & automated retraining.

---

## ✨ Features

- **State-of-the-art performance** on highly imbalanced data (0.172% fraud)
- Multiple imbalance handling strategies (Class Weights, SMOTE, Balanced RF, LightGBM `scale_pos_weight`)
- **SHAP interpretability** — explain *why* every transaction was flagged (critical for banks & regulators)
- Business-focused metrics (money saved, false alarms, chargeback impact)
- Full **feature engineering** (cyclical time, transaction velocity, interactions)
- Production-ready components:
  - FastAPI real-time scoring endpoint (< 10ms)
  - Model serialization & versioning
  - Drift detection (PSI)
  - Automated retraining pipeline
  - Monitoring dashboard skeleton
- Complete Jupyter-style walkthrough with explanations

---

## 📊 Dataset

**Credit Card Fraud Detection Dataset (ULB, 2013)**

- 284,807 transactions from European cardholders
- Only **492 fraudulent** transactions → **0.172% fraud rate** (extreme imbalance)
- 31 features: `Time`, `V1-V28` (PCA-transformed for privacy), `Amount`, `Class`

---

## 🚧 Key Challenges Solved

| Challenge              | Solution Implemented                          |
|------------------------|-----------------------------------------------|
| Extreme class imbalance| Class weights, SMOTE, LightGBM scale_pos_weight |
| Real-time requirement  | LightGBM (<10ms inference)                    |
| Regulatory need        | Full SHAP explanations + waterfall plots      |
| Evolving fraud patterns| PSI drift detection + automated retraining    |
| Business impact        | Custom cost-sensitive threshold selection     |

---

## 🛠️ Tech Stack

- **Core**: Python 3.11, pandas, NumPy, scikit-learn
- **ML**: LightGBM (recommended), RandomForest, XGBoost
- **Imbalance**: imbalanced-learn (SMOTE)
- **Explainability**: SHAP
- **Deployment**: FastAPI + Uvicorn
- **Serialization**: joblib + ONNX (optional)
- **Visualization**: Matplotlib, Seaborn
- **Monitoring**: Custom PSI drift + alerting skeleton

---

## 📥 Installation

```bash
# 1. Clone the repo
git clone https://github.com/yourusername/fraudguard.git
cd fraudguard

# 2. Create conda environment
conda create -n fraud_detection python=3.11 -y
conda activate fraud_detection

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download dataset
# Place creditcard.csv in the data/ folder
# (Download from: https://www.kaggle.com/mlg-ulb/creditcardfraud)
