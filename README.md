# 🎓 Behavioural Burnout Intelligence System

Early Detection of Academic Disengagement using Behavioural Analytics

---

## 📊 Dashboard Preview

### 🔎 System Overview

<p align="center">
  <img src="assets/dashboard_overview.png" width="900">
</p>

### 📈 Behavioural Timeline Monitoring

<p align="center">
  <img src="assets/dashboard_timeline.png" width="900">
</p>

## Problem Overview

Student burnout and academic disengagement often go unnoticed until performance significantly declines. 

This project builds a behavioural intelligence system that:
- Detects early warning signals
- Monitors engagement patterns over time
- Identifies behavioural instability
- Provides explainable risk predictions
- Suggests intervention strategies

---

## Dataset Description

Dataset Type: Synthetic

Why Synthetic?
- No real longitudinal behavioural dataset was available
- Privacy and access constraints
- Simulation allows controlled behavioural modelling

Number of Students: 5,000  
Number of Weeks: 16  
Total Records: 80,000 rows

Behavioural Archetypes:
- Stable
- Gradual Burnout
- Sudden Drop

---

## Feature Engineering

Raw Weekly Signals:
- login_count
- attendance_rate
- assignment_delay_days
- sentiment_score

Engineered Features:
- login_slope
- attendance_slope
- delay_slope
- sentiment_std
- shock_index
- consistency_score
- early_warning_flag

---

## Model Selection

Models Implemented:
1. Logistic Regression (interpretable baseline)
2. Random Forest (nonlinear deployment model)

Evaluation Metrics:
- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

The model avoids data leakage and uses engineered behavioural features.

---

## Explainability

SHAP was used to:
- Identify global behavioural drivers
- Explain individual student risk
- Rank feature impact

Key Insight:
Emotional volatility and behavioural instability are stronger predictors than single behavioural shocks.

---

## Dashboard Features

- Risk distribution overview
- Individual student profiling
- Cohort comparison (radar chart)
- 16-week behavioural timeline
- SHAP explanation
- Intervention recommendations

---

## How to Run

Install dependencies:

pip install -r requirements.txt

Run dashboard:

streamlit run app/dashboard_app.py