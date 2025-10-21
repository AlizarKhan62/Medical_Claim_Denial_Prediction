# src/preprocess.py
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def preprocess_data(df: pd.DataFrame):
    """Clean, encode, and scale the claim dataset."""
    print("Starting data preprocessing...")

    # Drop irrelevant columns
    drop_cols = ['Claim ID', 'Patient ID']
    df = df.drop(columns=[col for col in drop_cols if col in df.columns], errors='ignore')

    # Convert date
    if 'Date of Service' in df.columns:
        df['Date of Service'] = pd.to_datetime(df['Date of Service'], errors='coerce')
        df['Service_Year'] = df['Date of Service'].dt.year
        df['Service_Month'] = df['Date of Service'].dt.month
        df.drop('Date of Service', axis=1, inplace=True)

    # Encode binary column
    if 'Follow-up Required' in df.columns:
        df['Follow-up Required'] = df['Follow-up Required'].map({'Yes': 1, 'No': 0})

    # Encode categorical variables
    le = LabelEncoder()
    cat_cols = df.select_dtypes(include='object').columns
    for col in cat_cols:
        df[col] = le.fit_transform(df[col])

    # Separate features and target
    if 'Outcome' not in df.columns:
        raise ValueError("Target column 'Outcome' not found!")

    X = df.drop('Outcome', axis=1)
    y = df['Outcome']

    # Scale numerical features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("Data preprocessing completed.")
    return X_scaled, y, df
