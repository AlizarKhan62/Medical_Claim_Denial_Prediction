import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings("ignore")

def compare_models():
    # Load cleaned data
    data = pd.read_csv("data/processed/claims_clean.csv")
    target_col = "Claim Status"
    X = data.drop(columns=[target_col])
    y = data[target_col]

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale numeric features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Define models
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, random_state=42)
    }

    best_model = None
    best_acc = 0

    print("🔹 Training and Comparing Models...\n")

    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)
        acc = accuracy_score(y_test, preds)
        print(f"Model: {name}")
        print(f"Accuracy: {acc:.3f}")
        print(classification_report(y_test, preds))
        print("-" * 50)

        if acc > best_acc:
            best_acc = acc
            best_model = model
            best_name = name

    # Save the best model
    joblib.dump(best_model, f"models/best_model.pkl")
    print(f"\n✅ Best Model: {best_name} (Accuracy: {best_acc:.3f}) saved to models/best_model.pkl")

if __name__ == "__main__":
    compare_models()
