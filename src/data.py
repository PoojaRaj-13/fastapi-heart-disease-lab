import pandas as pd
from sklearn.model_selection import train_test_split

def load_data():
    """Load Heart Disease dataset from UCI repository."""
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    cols = [
        "age", "sex", "cp", "trestbps", "chol",
        "fbs", "restecg", "thalach", "exang",
        "oldpeak", "slope", "ca", "thal", "target"
    ]
    df = pd.read_csv(url, names=cols, na_values="?")
    df.dropna(inplace=True)

    # Binarize target: 0 = no disease, 1 = disease
    df["target"] = (df["target"] > 0).astype(int)

    X = df.drop("target", axis=1)
    y = df["target"]

    return train_test_split(X, y, test_size=0.2, random_state=42)