import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Load data
df = pd.read_csv("data/raw/synthetic_student_behaviour.csv")

student_features = []

for student_id, group in df.groupby("student_id"):
    
    group = group.sort_values("week")
    
    X = group["week"].values.reshape(-1, 1)
    
    def slope(y):
        model = LinearRegression()
        model.fit(X, y)
        return model.coef_[0]
    
    login_slope = slope(group["login_count"].values)
    attendance_slope = slope(group["attendance_rate"].values)
    delay_slope = slope(group["assignment_delay_days"].values)
    sentiment_slope = slope(group["sentiment_score"].values)
    
    login_std = np.std(group["login_count"])
    sentiment_std = np.std(group["sentiment_score"])
    
    # Shock index (max negative drop)
    login_diff = np.diff(group["login_count"])
    shock_index = np.min(login_diff)
    
    # Consistency score
    consistency_score = 1 / (1 + np.var(group["login_count"]))
    
    # Early warning (first 8 vs last 8 weeks)
    first_half = group[group["week"] <= 8]["login_count"].mean()
    second_half = group[group["week"] > 8]["login_count"].mean()
    
    early_warning_flag = 1 if (first_half - second_half) > 5 else 0
    
    student_features.append([
        student_id,
        login_slope,
        attendance_slope,
        delay_slope,
        sentiment_slope,
        login_std,
        sentiment_std,
        shock_index,
        consistency_score,
        early_warning_flag
    ])

features_df = pd.DataFrame(student_features, columns=[
    "student_id",
    "login_slope",
    "attendance_slope",
    "delay_slope",
    "sentiment_slope",
    "login_std",
    "sentiment_std",
    "shock_index",
    "consistency_score",
    "early_warning_flag"
])

features_df.to_csv("student_level_features.csv", index=False)

print("Feature engineering complete.")
print(features_df.head())