import pandas as pd
import shap
import joblib
import matplotlib.pyplot as plt
import numpy as np

# -----------------------------
# 1️⃣ Load Data & Model
# -----------------------------

df = pd.read_csv("data/processed/student_final_dataset.csv")

X = df[[
    "login_std",
    "sentiment_std",
    "shock_index",
    "consistency_score",
    "early_warning_flag"
]]

model = joblib.load("models/burnout_model.pkl")

# -----------------------------
# 2️⃣ SHAP Explainer
# -----------------------------

explainer = shap.TreeExplainer(model)
shap_values = explainer(X)

print("X shape:", X.shape)
print("Raw SHAP shape:", shap_values.values.shape)

vals = shap_values.values

# Your case: (samples, features, classes)
if len(vals.shape) == 3:
    shap_vals = vals[:, :, 1]  # select burnout class
    base_value = shap_values.base_values[:, 1]
else:
    shap_vals = vals
    base_value = shap_values.base_values

# -----------------------------
# 3️⃣ Global Summary Plot
# -----------------------------

plt.figure()
shap.summary_plot(shap_vals, X, show=False)
plt.title("SHAP Summary Plot - Burnout Prediction")
plt.tight_layout()
plt.savefig("reports/shap_summary_plot.png")
plt.close()

print("Global SHAP summary plot saved.")

# -----------------------------
# 4️⃣ Individual Student Plot
# -----------------------------

student_index = 10

plt.figure()
shap.force_plot(
    base_value[student_index],
    shap_vals[student_index],
    X.iloc[student_index],
    matplotlib=True,
    show=False
)

plt.title(f"SHAP Explanation for Student {student_index}")
plt.tight_layout()
plt.savefig("reports/shap_individual_student.png")
plt.close()

print("Individual SHAP explanation saved.")