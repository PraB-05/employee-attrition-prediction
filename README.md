# 📊 Employee Attrition Prediction

> Predicting which employees are likely to leave so HR teams can intervene proactively — before it's too late.

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-red)](https://employee-attrition-prediction-main.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.x-blue)]()
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange)]()

---

## 🎯 Business Problem

Every time a valuable employee leaves, companies spend an average of 6–9 months of that employee's salary on recruitment and training. This project builds a machine learning system to identify **which employees are at risk of leaving** — giving HR teams the information they need to intervene with the right people at the right time.

**Target audience:** HR teams, people analytics teams, business managers

---

## 📁 Dataset

- **Source:** IBM HR Analytics Employee Attrition Dataset (Kaggle)
- **Size:** 1,470 employees × 13 features
- **Target variable:** Attrition (Yes/No)
- **Class distribution:** 83.8% stayed, 16.2% left — heavily imbalanced

---

## 🔍 Key EDA Findings

| Finding | Detail |
|---|---|
| 💰 Income gap | Employees who left earned **30% less** on average (₹4,787 vs ₹6,832) |
| 🧑 Age | Quitters averaged **33 years** vs 37 for stayers — younger employees leave more |
| 📍 Distance | Quitters lived **further from office** on average (10.6 km vs 8.9 km) |
| 😔 Job satisfaction | Lower among quitters (2.47 vs 2.78 out of 4) |
| 🏢 Companies worked | Job hoppers leave more (2.94 vs 2.64 previous companies) |
| 💍 Marital status | Singles: 25.5% attrition — nearly double married employees (12.5%) |
| 🏬 Department | **Sales has highest attrition (20.6%)** — not R&D as raw counts suggested |
| 🎓 Education field | HR and Technical Degree holders show highest attrition (24–26%) |
| 📅 Age insight | Younger employees leave more — confirmed by feature importance |
| ⚠️ Accuracy trap | Random Forest: 83% accuracy but only 6% Recall — proving accuracy is misleading on imbalanced data |

> **Key insight:** Raw counts showed R&D as the highest attrition department. After normalizing by department size, Sales (20.6%) actually leads — demonstrating why percentage-based analysis is critical over absolute counts.

---

## ⚙️ Feature Engineering

Two new features were created with explicit business logic:

**1. IncomePerYear = MonthlyIncome / Age**
A 45-year-old earning ₹3,000/month is far more frustrated than a 25-year-old earning the same amount. This feature captures compensation relative to career stage — something raw salary alone cannot.

**2. SatisfactionScore = (JobSatisfaction + EnvironmentSatisfaction + WorkLifeBalance) / 3**
All three satisfaction metrics moved in the same direction. Combining them captures overall dissatisfaction as a single, more powerful signal.

✅ Both engineered features appeared in the **top 10 most important predictors** confirmed by Random Forest feature importance — validating the business logic behind creating them.

---

## 🤖 Model Comparison

| Model | Recall (quitters) | Precision | F1 | Accuracy |
|---|---|---|---|---|
| Logistic Regression (default) | 0.53 | 0.24 | 0.33 | 0.65 |
| Naive Bayes | 0.40 | 0.37 | 0.38 | 0.79 |
| SVM | 0.45 | 0.27 | 0.34 | 0.72 |
| Random Forest | 0.06 | 0.33 | 0.11 | 0.83 |
| **LR Tuned (C=0.01)** | **0.60** | **0.25** | **0.35** | **0.65** |
| **LR Tuned (5-fold CV)** | **0.68 ± 0.04** | - | - | - |

### Why Logistic Regression?

**Recall** was chosen as the primary metric because:
- Missing an employee who leaves = recruitment + training cost (expensive ❌)
- False alarm = one extra HR conversation (cheap ✅)

Despite Random Forest achieving 83% accuracy, its **6% Recall** made it practically useless for this business problem. Accuracy is a misleading metric on imbalanced datasets.

### Threshold Tuning

By adjusting the decision threshold from 0.5 → 0.3:

| Threshold | Recall | Precision |
|---|---|---|
| 0.50 (default) | 0.60 | 0.25 |
| 0.30 (tuned) | 0.91 | 0.19 |

For HR attrition — catching more quitters at the cost of more false alarms is the right tradeoff.

### Hyperparameter Tuning

GridSearchCV revealed **C=0.01** as optimal — strong regularization suits this small dataset (1,470 rows), reducing overfitting risk.

---

## 🚀 Streamlit App

An interactive web app allows HR teams to input employee details and get an instant attrition risk prediction with adjustable decision threshold.

**Features:**
- Employee detail input form
- Adjustable decision threshold slider
- High Risk / Low Risk prediction with probability score
- Real-time prediction

👉 **[Try the live app here](https://employee-attrition-prediction-main.streamlit.app/)**

---

## 🧰 Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3 |
| Data Analysis | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Machine Learning | Scikit-learn (LR, NB, SVM, RF) |
| Model Tuning | GridSearchCV, Pipeline, StandardScaler |
| Deployment | Streamlit, Streamlit Community Cloud |

---

## 📂 Project Structure

```
employee-attrition-prediction/
│
├── IBM_Attrition_prediction.ipynb   # Main notebook
├── app.py                           # Streamlit app
├── Final_Model_2.pkl               # Trained pipeline (scaler + LR)
├── README.md                        # This file
└── requirements.txt                 # Dependencies
```

---

## 🔮 Future Improvements

- Implement SMOTE for synthetic minority oversampling
- Try XGBoost and LightGBM
- Add SHAP values for individual prediction explanations
- Build department-level attrition dashboard
- Add model retraining pipeline with new data

---

## 💡 Key Learnings

1. **Accuracy is not enough** — always check class-specific metrics on imbalanced data
2. **Feature engineering matters** — both engineered features ranked in top 10 predictors
3. **Threshold tuning** is as important as model selection for imbalanced problems
4. **Business context drives metric choice** — Recall over Precision for HR attrition
5. **Pipeline prevents data leakage** — StandardScaler must be inside Pipeline for proper cross-validation
