# src/model_train.py
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from src.data_loader import load_data, save_data
from src.preprocess import preprocess_data

def train_model(data_path="data/raw/claim_data.csv", model_path="models/claim_model.pkl"):
    """Train model on preprocessed medical claim data."""
    df = load_data(data_path)
    X_scaled, y, processed_df = preprocess_data(df)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Train model
    print("Training Random Forest model...")
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Model trained. Accuracy: {acc:.3f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # Save model and processed data
    joblib.dump(model, model_path)
    save_data(processed_df, "data/processed/claims_clean.csv")
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    train_model()
