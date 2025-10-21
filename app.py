# app.py
"""
Streamlit Dashboard for Medical Claim Denial Prediction
This version fits LabelEncoders using your raw dataset (data/raw/claim_data.csv)
so uploaded files are encoded in the same way as training.
"""

import streamlit as st
import pandas as pd
import os
import joblib
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Claim Denial Predictor", layout="wide")

MODEL_PATH = "models/best_model.pkl"
RAW_DATA_PATH = "data/raw/claim_data.csv"
SAMPLE_PATH = "data/sample_claims.csv"

# -- helper functions -------------------------------------------------------
def load_model(path=MODEL_PATH):
    if not os.path.exists(path):
        return None
    return joblib.load(path)

def build_label_encoders(raw_df, categorical_cols):
    """
    Fit a LabelEncoder per categorical column using the raw training data.
    Returns a dict: {col_name: LabelEncoder()}
    """
    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        # fit on raw data (convert to string to avoid dtype issues)
        if col in raw_df.columns:
            le.fit(raw_df[col].astype(str))
        else:
            # Fit on an empty array to keep API consistent
            le.fit(pd.Series([], dtype="object").astype(str))
        encoders[col] = le
    return encoders

def transform_with_encoders(df, encoders, categorical_cols):
    """
    Transform df inplace using provided encoders.
    If an unseen label occurs, map it to -1.
    """
    df = df.copy()
    for col in categorical_cols:
        if col not in df.columns:
            continue
        le = encoders.get(col)
        if le is None:
            continue

        # create mapping dict for known classes
        mapping = {v: i for i, v in enumerate(le.classes_)}
        # map values; unknown -> -1
        df[col] = df[col].astype(str).map(lambda x: mapping.get(x, -1))
    return df

def validate_required_columns(df, required_cols):
    missing = [c for c in required_cols if c not in df.columns]
    return missing

# ----------------------------------------------------------------------------

st.title("🏥 Medical Claim Denial Predictor")
st.markdown(
    """
Upload a CSV with claim records or use the sample file. The model will predict the claim status outcome.
**Required columns** (case sensitive):  
`Provider ID`, `Billed Amount`, `Procedure Code`, `Diagnosis Code`, `Allowed Amount`, `Paid Amount`,
`Insurance Type`, `Claim Status`, `Reason Code`, `Follow-up Required`, `AR Status`, `Service_Year`, `Service_Month`
"""
)

# Load model
model = load_model()
if model is None:
    st.error("Model not found at `models/best_model.pkl`. Please run training (model_compare.py) first.")
else:
    st.success("✅ Model loaded.")

# Load raw dataset for encoders (used to keep encoding consistent)
raw_df = None
if os.path.exists(RAW_DATA_PATH):
    try:
        raw_df = pd.read_csv(RAW_DATA_PATH)
        st.info(f"Using raw dataset for encoders: `{RAW_DATA_PATH}` ({raw_df.shape[0]} rows).")
    except Exception as e:
        st.warning(f"Could not load raw training file `{RAW_DATA_PATH}` to build encoders: {e}")
else:
    st.warning(f"Raw training file `{RAW_DATA_PATH}` not found. Encoders will be fit from sample/unseen data (less reliable).")

# Define expected categorical columns (must match those used during training)
categorical_cols = [
    "Provider ID", "Procedure Code", "Diagnosis Code", "Insurance Type",
    "Claim Status", "Reason Code", "Follow-up Required", "AR Status"
]

# Build encoders from raw_df if available, else from sample (fallback)
if raw_df is not None:
    encoders = build_label_encoders(raw_df, categorical_cols)
else:
    # Fallback: build encoders from processed data or sample file if available
    fallback_df = None
    if os.path.exists(SAMPLE_PATH):
        try:
            fallback_df = pd.read_csv(SAMPLE_PATH)
        except Exception:
            fallback_df = None
    if fallback_df is not None:
        encoders = build_label_encoders(fallback_df, categorical_cols)
        st.info("Encoders built from sample file as fallback.")
    else:
        encoders = {col: None for col in categorical_cols}
        st.warning("No source to build encoders. Categorical encoding may be inconsistent.")

# File uploader
uploaded_file = st.file_uploader("Upload claims CSV (or leave empty to use sample data)", type=["csv"])

if uploaded_file:
    try:
        user_df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Unable to read uploaded file: {e}")
        st.stop()
else:
    # If sample exists, use it; otherwise create a small sample
    if os.path.exists(SAMPLE_PATH):
        user_df = pd.read_csv(SAMPLE_PATH)
        st.info(f"Using sample file: {SAMPLE_PATH}")
    else:
        # Create a small synthetic sample (same columns as our pipeline expects)
        user_df = pd.DataFrame({
            "Provider ID": ["P101", "P202"],
            "Billed Amount": [1200, 2500],
            "Procedure Code": ["99213", "99214"],
            "Diagnosis Code": ["D12", "E11"],
            "Allowed Amount": [800, 2000],
            "Paid Amount": [750, 1800],
            "Insurance Type": ["Private", "Medicare"],
            "Claim Status": ["Submitted", "Pending"],
            "Reason Code": ["R01", "R03"],
            "Follow-up Required": ["No", "Yes"],
            "AR Status": ["Open", "Closed"],
            "Service_Year": [2023, 2024],
            "Service_Month": [5, 8]
        })
        st.info("No upload or sample found — using small synthetic sample.")

st.write("### Preview (first 5 rows):")
st.dataframe(user_df.head())

# Validate required columns
required_cols = [
    "Provider ID", "Billed Amount", "Procedure Code", "Diagnosis Code",
    "Allowed Amount", "Paid Amount", "Insurance Type", "Claim Status",
    "Reason Code", "Follow-up Required", "AR Status", "Service_Year", "Service_Month"
]
missing = validate_required_columns(user_df, required_cols)
if missing:
    st.warning(f"The uploaded data is missing required columns: {missing}. Please upload a CSV with required columns.")
    st.stop()

# Encode categorical columns
user_df_encoded = transform_with_encoders(user_df, encoders, categorical_cols)

# Ensure numeric columns are numeric
num_cols = ["Billed Amount", "Allowed Amount", "Paid Amount", "Service_Year", "Service_Month"]
for c in num_cols:
    user_df_encoded[c] = pd.to_numeric(user_df_encoded[c], errors="coerce").fillna(0)

# Final features to feed model — match training pipeline's feature ordering if possible
# By default we feed the entire dataframe (except maybe an ID column)
feature_cols = [c for c in user_df_encoded.columns if c not in ("Claim ID", "Patient ID")]
X = user_df_encoded[feature_cols]

st.write("### Features used for prediction (first 5 rows):")
st.dataframe(X.head())

# Predict button
if st.button("🔮 Predict Claim Outcomes"):
    if model is None:
        st.error("Model is not available.")
    else:
        try:
            preds = model.predict(X)
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.stop()

        user_df["Predicted_Outcome"] = preds
        st.success("✅ Predictions complete!")
        st.write("### Prediction Results:")
        st.dataframe(user_df[["Provider ID", "Billed Amount", "Insurance Type", "Predicted_Outcome"]])

        # Save & download
        os.makedirs("results", exist_ok=True)
        out_path = "results/predictions_streamlit.csv"
        user_df.to_csv(out_path, index=False)
        st.download_button(
            label="⬇️ Download Predictions (CSV)",
            data=user_df.to_csv(index=False).encode("utf-8"),
            file_name="predictions_streamlit.csv",
            mime="text/csv"
        )

        # Simple visualization of predicted class counts
        st.subheader("📊 Predicted Outcome Distribution")
        counts = pd.Series(preds).value_counts().sort_index()
        chart_df = pd.DataFrame({"Outcome": counts.index.astype(str), "Count": counts.values})
        st.bar_chart(chart_df.set_index("Outcome"))
