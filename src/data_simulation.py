import numpy as np
import pandas as pd

np.random.seed(42)

# PARAMETERS
N_STUDENTS = 5000
N_WEEKS = 16

students = []
archetypes = []

# Assign behavioural archetypes
for i in range(N_STUDENTS):
    archetype = np.random.choice(
        ["stable", "gradual_burnout", "sudden_drop"],
        p=[0.6, 0.3, 0.1]
    )
    archetypes.append(archetype)

# Generate weekly behavioural data
for student_id in range(N_STUDENTS):

    archetype = archetypes[student_id]

    # Base levels
    base_login = np.random.randint(20, 35)
    base_attendance = np.random.uniform(0.75, 1.0)
    base_delay = np.random.uniform(0, 1)
    base_sentiment = np.random.uniform(0.3, 0.8)

    # Softer decline rates (more realistic)
    decline_rate = np.random.uniform(0.15, 0.5)
    delay_growth = np.random.uniform(0.05, 0.25)

    # Some students partially recover (5%)
    recovery_student = np.random.rand() < 0.05

    for week in range(1, N_WEEKS + 1):

        noise_login = np.random.normal(0, 2)
        noise_attendance = np.random.normal(0, 0.03)
        noise_delay = np.random.normal(0, 0.3)
        noise_sentiment = np.random.normal(0, 0.05)

        # ------------------------
        # STABLE STUDENTS
        # ------------------------
        if archetype == "stable":
            minor_drift = np.random.uniform(-0.2, 0.1)

            login = base_login + minor_drift * week + noise_login
            attendance = base_attendance + np.random.uniform(-0.005, 0.005) * week + noise_attendance
            delay = base_delay + np.random.uniform(-0.05, 0.05) * week + noise_delay
            sentiment = base_sentiment + np.random.uniform(-0.01, 0.01) * week + noise_sentiment

        # ------------------------
        # GRADUAL BURNOUT
        # ------------------------
        elif archetype == "gradual_burnout":

            # Recovery effect in later weeks
            recovery_adjustment = 0
            if recovery_student and week > 10:
                recovery_adjustment = 0.3 * (week - 10)

            login = base_login - decline_rate * week + recovery_adjustment + noise_login
            attendance = base_attendance - 0.015 * week + noise_attendance
            delay = base_delay + delay_growth * week + noise_delay
            sentiment = base_sentiment - 0.02 * week + noise_sentiment

        # ------------------------
        # SUDDEN DROP
        # ------------------------
        elif archetype == "sudden_drop":

            if week <= 8:
                login = base_login + noise_login
                attendance = base_attendance + noise_attendance
                delay = base_delay + noise_delay
                sentiment = base_sentiment + noise_sentiment
            else:
                shock_magnitude = np.random.uniform(5, 10)

                login = base_login - shock_magnitude - decline_rate * (week - 8) + noise_login
                attendance = base_attendance - 0.10 + noise_attendance
                delay = base_delay + 1.5 + delay_growth * (week - 8) + noise_delay
                sentiment = base_sentiment - 0.2 + noise_sentiment

        students.append([
            student_id,
            week,
            max(login, 0),
            min(max(attendance, 0), 1),
            max(delay, 0),
            min(max(sentiment, -1), 1),
            archetype
        ])

# Create DataFrame
df = pd.DataFrame(students, columns=[
    "student_id",
    "week",
    "login_count",
    "attendance_rate",
    "assignment_delay_days",
    "sentiment_score",
    "archetype"
])

# Save dataset
df.to_csv("data/raw/synthetic_student_behaviour.csv", index=False)

print("Realistic synthetic dataset generated successfully.")
print(df.head())