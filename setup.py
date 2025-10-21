import os

# === Define all folders ===
folders = [
    "data/raw",
    "data/processed",
    "notebooks",
    "models",
    "src",
    "app",
    "logs",
    "reports",
    "tests"
]

# === Define all files (with minimal starter content) ===
files = {
    "README.md": "# Medical Claim Denial Prediction Project\n\nThis project predicts whether a healthcare claim will be approved or denied using machine learning.",
    ".gitignore": "__pycache__/\n*.pkl\n*.csv\n*.log\n.env\n",
    "src/__init__.py": "",
    "src/data_loader.py": "# Handles data loading and saving\nimport pandas as pd\n\ndef load_data(path):\n    return pd.read_csv(path)",
    "src/preprocess.py": "# Data cleaning and preprocessing functions\ndef preprocess(df):\n    # Add preprocessing logic here\n    return df",
    "src/model_train.py": "# Model training script\n\ndef train_model():\n    pass",
    "src/evaluate.py": "# Model evaluation script\ndef evaluate_model():\n    pass",
    "src/utils.py": "# Helper functions\nimport logging\n\ndef setup_logger(log_path='logs/training.log'):\n    logging.basicConfig(filename=log_path, level=logging.INFO)\n    return logging.getLogger(__name__)",
    "app/app.py": "import streamlit as st\n\nst.title('Medical Claim Denial Prediction')\nst.write('Upload patient and billing details to predict claim denial.')",
    "app/requirements.txt": "pandas\nnumpy\nscikit-learn\nxgboost\nimbalanced-learn\nmatplotlib\nseaborn\nshap\nstreamlit\njoblib",
    "reports/project_report.md": "# Project Report\n\n## Objective\nPredict insurance claim denials using machine learning.",
    "tests/test_data.py": "def test_data_loading():\n    assert True",
    "tests/test_model.py": "def test_model_training():\n    assert True"
}

# === Function to create folder & files ===
def create_structure():
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
        print(f"✅ Created folder: {folder}")

    for file, content in files.items():
        if not os.path.exists(file):
            with open(file, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"📄 Created file: {file}")
        else:
            print(f"⚠️ File already exists: {file}")

if __name__ == "__main__":
    print("\n🚀 Setting up Medical Claim Denial Prediction project...\n")
    create_structure()
    print("\n✅ Project structure ready! Open it in VS Code and start building.\n")
