import pandas as pd
from sklearn.preprocessing import LabelEncoder

def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    le = LabelEncoder()

    if "education" in df.columns:
        df["education"] = df["education"].fillna("Unknown")  # fixed indentation
        df["education"] = le.fit_transform(df["education"])

    if "self_employed" in df.columns:
        df["self_employed"] = df["self_employed"].fillna("No")  # fixed indentation
        df["self_employed"] = le.fit_transform(df["self_employed"])

    return df  # return inside the function

def encode_target(df: pd.DataFrame) -> pd.DataFrame:
    # Strip spaces first
    if "loan_status" in df.columns:
        df["loan_status"] = df["loan_status"].str.strip()
        df["loan_status"] = df["loan_status"].map({
            "Approved": 1,
            "Rejected": 0
        })

    return df  # return inside the function
