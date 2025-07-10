import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


def load_and_clean_data(filepath):
    """
    Load and clean the flight delay dataset.
    - Remove cancelled and diverted flights
    - Convert time columns
    - Handle missing values
    - Drop unnecessary columns
    """
    df = pd.read_csv(filepath)

    # Drop cancelled and diverted flights
    df = df[(df['CANCELLED'] == 0) & (df['DIVERTED'] == 0)].copy()
    df.drop(columns=['CANCELLED', 'DIVERTED', 'CANCELLATION_CODE'], inplace=True, errors='ignore')

    # Fill missing delay components with 0
    delay_cols = [
        'DELAY_DUE_CARRIER', 'DELAY_DUE_WEATHER',
        'DELAY_DUE_NAS', 'DELAY_DUE_SECURITY', 'DELAY_DUE_LATE_AIRCRAFT'
    ]
    df[delay_cols] = df[delay_cols].fillna(0)

    # Convert FL_DATE to datetime and extract day of week
    df['FL_DATE'] = pd.to_datetime(df['FL_DATE'], errors='coerce')
    df['DAY_OF_WEEK'] = df['FL_DATE'].dt.dayofweek
    df.drop(columns=['FL_DATE'], inplace=True)

    # Convert CRS_DEP_TIME and CRS_ARR_TIME to hour
    df['CRS_DEP_HOUR'] = df['CRS_DEP_TIME'].astype(str).str.zfill(4).str[:2].astype(int)
    df['CRS_ARR_HOUR'] = df['CRS_ARR_TIME'].astype(str).str.zfill(4).str[:2].astype(int)
    df.drop(columns=['CRS_DEP_TIME', 'CRS_ARR_TIME'], inplace=True)

    # One-hot encode categorical columns
    df = pd.get_dummies(df, columns=['AIRLINE', 'ORIGIN', 'DEST','AIRLINE_DOT', 'AIRLINE_CODE', 'ORIGIN_CITY', 'DEST_CITY'], drop_first=True)

    # Drop unused columns
    df.drop(columns=['ELAPSED_TIME', 'ARR_TIME', 'DEP_TIME', 'CANCELLED', 'DIVERTED', 'DOT_CODE'], inplace=True, errors='ignore')

    # Drop rows with missing target
    df = df[df['ARR_DELAY'].notnull()]

    return df


def get_feature_target_split(df):
    """
    Split dataframe into features and target
    """
    X = df.drop(columns=['ARR_DELAY'])
    y = df['ARR_DELAY']
    return X, y


def get_preprocessor(numeric_features):
    """
    Return a ColumnTransformer for preprocessing
    - Scales numeric features
    - Leaves categorical features (already one-hot encoded)
    """
    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(), numeric_features)
    ], remainder='passthrough')

    return preprocessor
