import pandas as pd
import os
from .encoding import encode_features

def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean column names: strip spaces, lowercase, replace spaces with underscores
    """
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )
    return df

def save_preprocessed_data(df: pd.DataFrame, path: str):
    """
    Save dataframe to CSV, creating directories if needed
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)

def preprocess_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean, encode features, and drop unnecessary columns like loan_id.
    Returns feature set ready for training or prediction.
    """
    df = clean_columns(df)
    df = encode_features(df)
    df = df.drop(columns=["loan_id"], errors="ignore")  # drop loan_id automatically
    return df
