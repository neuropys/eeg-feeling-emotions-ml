import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split


def load_emotions_dataset(csv_path: str):
    """
    Load the EEG Brainwave 'emotions.csv' dataset from Kaggle.

    Parameters
    ----------
    csv_path : str
        Path to emotions.csv

    Returns
    -------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features).
    y : np.ndarray
        Encoded labels (0, 1, 2).
    label_encoder : LabelEncoder
        Fitted encoder to map integers back to class names.
    feature_names : list of str
        Names of feature columns.
    """
    df = pd.read_csv(csv_path)

    if "label" not in df.columns:
        raise ValueError("Expected a 'label' column in emotions.csv")

    # Separate features and labels
    y_str = df["label"].values                # string labels
    X = df.drop(columns=["label"]).values     # all feature columns
    feature_names = df.drop(columns=["label"]).columns.tolist()

    # Encode string labels to integers
    le = LabelEncoder()
    y = le.fit_transform(y_str)  # e.g., NEGATIVE->0, NEUTRAL->1, POSITIVE->2

    return X, y, le, feature_names


def train_test_split_scaled(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42
):
    """
    Split into train/test and standardize features.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y  # keep class balance
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler
