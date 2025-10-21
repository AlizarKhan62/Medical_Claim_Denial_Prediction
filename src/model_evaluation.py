import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

def evaluate_models():
    # Load cleaned, processed data
    data = pd.read_csv("data/processed/claims_clean.csv")
    print(f"✅ Data loaded successfully with {data.shape[0]} rows and {data.shape[1]} columns.")
    
    print("📊 Available columns:", list(data.columns))

    # Define target column
    target_col = "Claim Status"

    # Separate features and target
    X = data.drop(columns=[target_col])
    y = data[target_col]

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Load saved Random Forest model
    model = joblib.load("models/claim_model.pkl")

    # Predict
    y_pred = model.predict(X_test)

    # Evaluate
    print("\n🔹 Random Forest Model Evaluation:")
    print("Accuracy:", round(accuracy_score(y_test, y_pred), 3))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Confusion matrix
    plt.figure(figsize=(6, 4))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix - Random Forest Model")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

if __name__ == "__main__":
    evaluate_models()
