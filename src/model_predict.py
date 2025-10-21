"""
model_predict.py
----------------
This script loads the best trained model (from models/best_model.pkl)
and predicts outcomes for new unseen medical claim data.
"""

import os
import pickle
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Paths
model_path = "models/best_model.pkl"
sample_data_path = "data/sample_claims.csv"
output_path = "results/predictions_output.csv"

# Ensure folders exist
os.makedirs("results", exist_ok=True)

# ---------------------------
# 1. Load Trained Model
# ---------------------------
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Trained model not found at {model_path}. Please run model_compare.py first.")

print("Loading trained model...")

model = joblib.load(model_path)


# ---------------------------
# 2. Load Sample Input Data
# ---------------------------
if not os.path.exists(sample_data_path):
    print("Sample file not found. Creating synthetic data for prediction...")

    # Synthetic test data (you can replace this later)
    sample_data = pd.DataFrame({
        "Provider ID": ["P101", "P202", "P303", "P404", "P505"],
        "Billed Amount": [1200, 2500, 900, 3400, 1800],
        "Procedure Code": ["99213", "99214", "99212", "99215", "99213"],
        "Diagnosis Code": ["D12", "E11", "I10", "C34", "J20"],
        "Allowed Amount": [800, 2000, 700, 2900, 1600],
        "Paid Amount": [750, 1800, 600, 2500, 1500],
        "Insurance Type": ["Private", "Medicare", "Medicaid", "Private", "Medicare"],
        "Claim Status": ["Submitted", "Pending", "Submitted", "Denied", "Submitted"],
        "Reason Code": ["R01", "R03", "R02", "R04", "R01"],
        "Follow-up Required": ["No", "Yes", "No", "Yes", "No"],
        "AR Status": ["Open", "Closed", "Open", "Closed", "Open"],
        "Service_Year": [2023, 2024, 2024, 2025, 2023],
        "Service_Month": [5, 8, 11, 2, 7]
    })
    sample_data.to_csv(sample_data_path, index=False)
    print(f"Synthetic data saved to {sample_data_path}")
else:
    sample_data = pd.read_csv(sample_data_path)

print(f"Sample data loaded with {sample_data.shape[0]} records.")

# ---------------------------
# 3. Preprocess Data
# ---------------------------
# Encode categorical columns similar to training
categorical_cols = ["Provider ID", "Procedure Code", "Diagnosis Code", "Insurance Type",
                    "Claim Status", "Reason Code", "Follow-up Required", "AR Status"]

label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    sample_data[col] = le.fit_transform(sample_data[col].astype(str))
    label_encoders[col] = le

# ---------------------------
# 4. Make Predictions
# ---------------------------
print("\nMaking Predictions...")
preds = model.predict(sample_data)

sample_data["Predicted_Outcome"] = preds
print("Predictions complete!")

# ---------------------------
# 5. Save Results
# ---------------------------
os.makedirs("results", exist_ok=True)
sample_data.to_csv(output_path, index=False)
print(f"Predictions saved to {output_path}")

# ---------------------------
# 6. Display Results
# ---------------------------
print("\nPrediction Summary:")
print(sample_data[["Provider ID", "Billed Amount", "Insurance Type", "Predicted_Outcome"]])
