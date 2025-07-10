# --- preprocessing.py ---
import pandas as pd
import numpy as np
import category_encoders as ce
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder

def load_and_clean_data(path):
    df = pd.read_csv(path)

    delay_cols = ['DELAY_DUE_CARRIER', 'DELAY_DUE_WEATHER', 
                  'DELAY_DUE_NAS', 'DELAY_DUE_SECURITY', 
                  'DELAY_DUE_LATE_AIRCRAFT']
    df[delay_cols] = df[delay_cols].fillna(0)

    df = remove_outliers_iqr(df)

    for col in ['CRS_DEP_TIME', 'CRS_ARR_TIME', 'WHEELS_OFF', 'WHEELS_ON', 'DEP_TIME', 'ARR_TIME']:
        if col in df.columns:
            df[[f'{col}_HOUR_SIN', f'{col}_HOUR_COS', f'{col}_MINUTE_SIN', f'{col}_MINUTE_COS']] = df[col].apply(
                lambda x: pd.Series(cyclical_encoding(x)))

    df = split_and_overwrite_city_state(df, ['DEST_CITY', 'ORIGIN_CITY'])
    df = add_weekday_weekend_columns(df, ['FL_DATE'])

    df = encode_features(df)

    drop_cols = [
        'AIRLINE_DOT', 'AIRLINE_CODE', 'DOT_CODE', 'CANCELLED', 'CANCELLATION_CODE', 'DIVERTED',
        'DELAY_DUE_CARRIER', 'DELAY_DUE_WEATHER', 'DELAY_DUE_NAS', 'DELAY_DUE_SECURITY', 'DELAY_DUE_LATE_AIRCRAFT'
    ]
    df = df.drop(columns=[col for col in drop_cols if col in df.columns])
    #Drop datetime columns before returning
    df = df.select_dtypes(exclude=['datetime64[ns]'])

    # Drop any non-numeric (string/object) columns that remain
    non_numeric_cols = df.select_dtypes(include=['object']).columns.tolist()
    if non_numeric_cols:
        print(f"Dropping non-numeric columns: {non_numeric_cols}")
        df = df.drop(columns=non_numeric_cols)

    return df

def remove_outliers_iqr(df):
    for col in df.select_dtypes(include='number').columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        df = df[(df[col] >= Q1 - 1.5 * IQR) & (df[col] <= Q3 + 1.5 * IQR)]
    return df

def cyclical_encoding(t):
    if pd.isnull(t): return np.nan, np.nan, np.nan, np.nan
    t = int(t)
    hour, minute = t // 100, t % 100
    return np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24), np.sin(2 * np.pi * minute / 60), np.cos(2 * np.pi * minute / 60)

def split_and_overwrite_city_state(df, cols):
    for col in cols:
        base = col.split('_')[0].lower()
        split = df[col].str.split(',', expand=True)
        df[col] = split[0].str.strip()
        df[f'{base}_state'] = split[1].str.strip() if split.shape[1] > 1 else None
    return df

def add_weekday_weekend_columns(df, cols):
    for col in cols:
        df[col] = pd.to_datetime(df[col], errors='coerce')
        df[f'{col}_day_name'] = df[col].dt.day_name()
        df[f'{col}_is_weekend'] = df[col].dt.dayofweek.isin([5, 6])
    return df

def encode_features(df):
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    oe = OrdinalEncoder(categories=[day_order])
    df['FL_DATE_day_name_encoded'] = oe.fit_transform(df[['FL_DATE_day_name']])

    ohe = OneHotEncoder(sparse_output=False)
    airline_encoded = ohe.fit_transform(df[['AIRLINE']])
    df[ohe.get_feature_names_out(['AIRLINE'])] = airline_encoded
    
    bin_enc = ce.BinaryEncoder(cols=['FL_NUMBER', 'ORIGIN', 'ORIGIN_CITY', 'dest_state', 'origin_state', 'DEST', 'DEST_CITY'])
    df = bin_enc.fit_transform(df)

    df = df.drop(columns=['FL_DATE_day_name', 'AIRLINE'])
    return df