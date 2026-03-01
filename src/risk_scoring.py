import pandas as pd
import numpy as np

# -----------------------------
# 1️⃣ Load Data
# -----------------------------

# Student-level engineered features
features_df = pd.read_csv("data/processed/student_level_features.csv")

# Raw data (to extract archetype ground truth)
raw_df = pd.read_csv("data/raw/synthetic_student_behaviour.csv")

# -----------------------------
# 2️⃣ Define Ground Truth
# -----------------------------

# Extract one archetype per student
archetype_map = raw_df.groupby("student_id")["archetype"].first().reset_index()

# Merge with features
df = features_df.merge(archetype_map, on="student_id")

# Burnout ground truth:
# Stable = 0
# Gradual Burnout + Sudden Drop = 1
df["burnout_label"] = df["archetype"].apply(
    lambda x: 0 if x == "stable" else 1
)

# -----------------------------
# 3️⃣ Risk Score (0–100)
# -----------------------------

def normalize(series):
    return (series - series.min()) / (series.max() - series.min())

login_risk = normalize(-df["login_slope"])
attendance_risk = normalize(-df["attendance_slope"])
delay_risk = normalize(df["delay_slope"])
volatility_risk = normalize(df["login_std"])
shock_risk = normalize(-df["shock_index"])

df["risk_score"] = (
    0.30 * login_risk +
    0.25 * attendance_risk +
    0.20 * delay_risk +
    0.15 * volatility_risk +
    0.10 * shock_risk
) * 100

# Risk Level Categories
df["risk_level"] = pd.cut(
    df["risk_score"],
    bins=[0, 40, 70, 100],
    labels=["Low", "Medium", "High"]
)

# -----------------------------
# 4️⃣ Save Final Dataset
# -----------------------------

df.to_csv("data/processed/student_final_dataset.csv", index=False)

print("Updated burnout labels using archetype.")
print(df["burnout_label"].value_counts())
print(df.head())