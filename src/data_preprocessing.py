import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split


def load_data(file_path):
    """Load the insurance dataset from a CSV file."""
    return pd.read_csv(file_path)


def create_synthetic_features(df):
    """Add auto insurance style features to the dataset."""
    df = df.copy()
    np.random.seed(42)

    df['vehicle_age'] = np.clip((df['age'] * 0.15 + np.random.normal(4, 3, len(df))).round(), 0, 12).astype(int)
    df['annual_mileage'] = np.clip(
        (12000 + (df['vehicle_age'] * 300) - (df['age'] * 15) + np.random.normal(0, 1700, len(df))).round(),
        5000,
        25000,
    ).astype(int)

    df['accident_history'] = np.clip(
        np.random.poisson(0.5, len(df)) + (df['smoker'] == 'yes').astype(int),
        0,
        4,
    ).astype(int)

    df['credit_score'] = np.clip(
        (720 - (df['bmi'] - 26) * 3 - df['accident_history'] * 12 + np.random.normal(0, 35, len(df))).round(),
        300,
        850,
    ).astype(int)

    df['driving_experience'] = np.clip((df['age'] - 16), 0, 50).astype(int)

    df['premium'] = (
        df['charges'] * 0.9
        + df['vehicle_age'] * 85
        + df['accident_history'] * 1100
        - df['credit_score'] * 3
        + np.random.normal(0, 900, len(df))
    ).round(2)
    df['premium'] = df['premium'].clip(lower=1000)

    risk_score = (
        (df['charges'] > df['charges'].quantile(0.65)).astype(int)
        + (df['accident_history'] >= 1).astype(int)
        + (df['smoker'] == 'yes').astype(int)
    )
    df['claim_risk'] = (risk_score >= 2).astype(int)

    return df


def generate_dataset(source_csv, output_csv):
    """Build the cleaned insurance dataset and save it for model training."""
    source_path = Path(source_csv)
    output_path = Path(output_csv)

    df = pd.read_csv(source_path)
    df = create_synthetic_features(df)

    # Keep features needed for the project and export a clean CSV file
    columns = [
        'age',
        'sex',
        'bmi',
        'children',
        'smoker',
        'region',
        'vehicle_age',
        'annual_mileage',
        'accident_history',
        'credit_score',
        'driving_experience',
        'claim_risk',
        'premium',
    ]
    df = df[columns]
    df.to_csv(output_path, index=False)
    return df


def create_data_if_missing(source_csv, output_csv):
    """Create the dataset file if it does not already exist."""
    source_path = Path(source_csv)
    output_path = Path(output_csv)

    if output_path.exists():
        return load_data(output_path)

    if not source_path.exists():
        raise FileNotFoundError(
            f"Source data file not found: {source_path}. Place a raw insurance CSV here."
        )

    return generate_dataset(source_path, output_path)


def clean_data(df):
    """Clean the dataset and prepare it for modeling."""
    df = df.copy()
    df = df.drop_duplicates()
    df = df.dropna()
    return df


def encode_features(df):
    """Encode categorical features into numeric values."""
    df = df.copy()
    df['sex'] = df['sex'].map({'female': 0, 'male': 1})
    df['smoker'] = df['smoker'].map({'no': 0, 'yes': 1})
    region_dummies = pd.get_dummies(df['region'], prefix='region', drop_first=True)
    df = pd.concat([df.drop(columns=['region']), region_dummies], axis=1)
    return df


def prepare_features(df):
    """Prepare features and labels for classification and regression."""
    df = clean_data(df)
    df = encode_features(df)

    feature_columns = [
        'age',
        'sex',
        'bmi',
        'children',
        'smoker',
        'vehicle_age',
        'annual_mileage',
        'accident_history',
        'credit_score',
        'driving_experience',
        'region_northwest',
        'region_southeast',
        'region_southwest',
    ]

    X = df[feature_columns]
    y_class = df['claim_risk']
    y_reg = df['premium']
    return X, y_class, y_reg


def split_data(X, y, test_size=0.2, random_state=42):
    """Split the dataset into train and test sets."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
