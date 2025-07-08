import pandas as pd
import numpy as np
import os
import logging
import math

import category_encoders as ce
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder

# Configure a logger for this module
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

class DataPreprocessor:
    """
    Handles data cleaning, missing value imputation, outlier treatment,
    feature engineering, and encoding for the flight delay dataset.
    """
    def __init__(self):
        logger.info("DataPreprocessor initialized.")

    def handle_missing_values(self, df: pd.DataFrame, delay_cols: list) -> pd.DataFrame:
        """
        Imputes missing values for specified delay columns with 0.

        Args:
            df (pd.DataFrame): The input DataFrame.
            delay_cols (list): A list of column names representing delay durations.

        Returns:
            pd.DataFrame: The DataFrame with missing delay values imputed.
        """
        df_copy = df.copy()
        logger.info(f"Handling missing values for delay columns: {delay_cols}")
        for col in delay_cols:
            if col in df_copy.columns:
                missing_count = df_copy[col].isnull().sum()
                if missing_count > 0:
                    df_copy[col] = df_copy[col].fillna(0)
                    logger.info(f"Imputed {missing_count} missing values in '{col}' with 0.")
            else:
                logger.warning(f"Delay column '{col}' not found in DataFrame. Skipping imputation for this column.")
        return df_copy

    def remove_outliers_iqr(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Removes outliers from numerical columns using the IQR method.

        Args:
            df (pd.DataFrame): The input DataFrame.

        Returns:
            pd.DataFrame: The DataFrame with outliers removed.
        """
        df_cleaned = df.copy()
        initial_rows = len(df_cleaned)
        
        numeric_cols = df_cleaned.select_dtypes(include='number').columns
        logger.info(f"Detecting and treating outliers using IQR method for {len(numeric_cols)} numerical columns.")

        # Exclude specific columns from outlier removal if they are targets or known to have extreme but valid values
        # For instance, if 'FLIGHT_STATUS' (your target) is numeric, don't remove outliers from it.
        # Ensure 'FLIGHT_STATUS' is not in numeric_cols if it's binary (0/1)
        cols_to_exclude_from_outlier = ['FLIGHT_STATUS'] # Add other target or identifier columns if needed
        numeric_cols = [col for col in numeric_cols if col not in cols_to_exclude_from_outlier]


        for col in numeric_cols:
            Q1 = df_cleaned[col].quantile(0.25)
            Q3 = df_cleaned[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Identify outliers in the current column
            outliers_count = len(df_cleaned[(df_cleaned[col] < lower_bound) | (df_cleaned[col] > upper_bound)])
            if outliers_count > 0:
                logger.debug(f"Column '{col}': Found {outliers_count} outliers (IQR method). Bounds: [{lower_bound:.2f}, {upper_bound:.2f}]")
                
            # Filter rows where values are within bounds
            df_cleaned = df_cleaned[(df_cleaned[col] >= lower_bound) & (df_cleaned[col] <= upper_bound)]
        
        rows_removed = initial_rows - len(df_cleaned)
        logger.info(f"Removed {rows_removed} rows containing outliers across numerical columns.")
        return df_cleaned

    def cyclical_encoding(self, t):
        """
        Helper function: Returns sine and cosine transformations for time in HHMM format.
        """
        if pd.isnull(t) or not isinstance(t, (int, float)): # Check for NaN and non-numeric types
            return np.nan, np.nan, np.nan, np.nan # Return NaN for all four outputs if input is NaN or invalid
        else:
            t = int(t) # Ensure integer for hour/minute separation
            hour = t // 100
            minute = t % 100
            
            # Normalize hour and minute
            hour_norm = hour / 24
            minute_norm = minute / 60
            
            hour_sin = np.sin(2 * np.pi * hour_norm)
            hour_cos = np.cos(2 * np.pi * hour_norm)
            minute_sin = np.sin(2 * np.pi * minute_norm)
            minute_cos = np.cos(2 * np.pi * minute_norm)
            
            return hour_sin, hour_cos, minute_sin, minute_cos

    def apply_cyclical_encoding(self, df: pd.DataFrame, time_columns: list) -> pd.DataFrame:
        """
        Applies cyclical encoding (sine/cosine transformation) to specified time columns.

        Args:
            df (pd.DataFrame): The input DataFrame.
            time_columns (list): List of column names in HHMM format to encode.

        Returns:
            pd.DataFrame: The DataFrame with new cyclical features.
        """
        df_copy = df.copy()
        logger.info(f"Applying cyclical encoding for time columns: {time_columns}")
        for col in time_columns:
            if col in df_copy.columns:
                # Ensure the column is numeric before applying
                if not pd.api.types.is_numeric_dtype(df_copy[col]):
                    logger.warning(f"Column '{col}' is not numeric. Attempting to convert before cyclical encoding.")
                    df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce') # Coerce non-numeric to NaN

                # Apply cyclical encoding
                # Using apply with result_type='expand' is often more efficient for multiple returns
                encoded_features = df_copy[col].apply(
                    lambda x: pd.Series(self.cyclical_encoding(x),
                                        index=[f'{col}_HOUR_SIN', f'{col}_HOUR_COS', f'{col}_MINUTE_SIN', f'{col}_MINUTE_COS'])
                )
                # Join the new features back to the original DataFrame, aligning by index
                df_copy = df_copy.join(encoded_features)
                logger.info(f"Created cyclical features for '{col}'. New columns: {', '.join(encoded_features.columns)}")
            else:
                logger.warning(f"Time column '{col}' not found in DataFrame. Skipping cyclical encoding for this column.")
        return df_copy

    def split_city_state(self, df: pd.DataFrame, column_names: list) -> pd.DataFrame:
        """
        Splits columns like 'DEST_CITY' into 'dest_city' and 'dest_state'.
        The original column is overwritten with the city name, and a new state column is added.

        Args:
            df (pd.DataFrame): The input DataFrame.
            column_names (list): List of column names (e.g., ['DEST_CITY', 'ORIGIN_CITY']) to split.

        Returns:
            pd.DataFrame: The DataFrame with split city/state information.
        """
        df_copy = df.copy()
        logger.info(f"Splitting city and state from columns: {column_names}")
        for column_name in column_names:
            if column_name in df_copy.columns:
                base_name = column_name.split('_')[0].lower() # e.g., 'dest' or 'origin'
                
                # Split city and state (e.g., "New York, NY" → ["New York", "NY"])
                # r'\s*,\s*' handles optional spaces around comma
                split_result = df_copy[column_name].astype(str).str.split(r'\s*,\s*', n=1, expand=True)
                
                # Overwrite original column with clean city name
                df_copy[column_name] = split_result[0].str.strip()
                
                # Add new state column
                df_copy[f"{base_name}_state"] = split_result[1].str.strip()
                logger.info(f"Split '{column_name}' into '{column_name}' (city) and '{base_name}_state'.")
            else:
                logger.warning(f"Column '{column_name}' not found for city/state splitting.")
        return df_copy

    def add_weekday_weekend_columns(self, df: pd.DataFrame, date_columns: list) -> pd.DataFrame:
        """
        Adds weekday name and weekend boolean columns for each specified date column.

        Args:
            df (pd.DataFrame): Input DataFrame.
            date_columns (list): List of column names containing dates.

        Returns:
            pd.DataFrame: DataFrame with added day-related columns.
        """
        df_copy = df.copy()
        logger.info(f"Adding weekday/weekend columns for date columns: {date_columns}")
        
        for col in date_columns:
            if col in df_copy.columns:
                # Ensure column is in datetime format
                if not pd.api.types.is_datetime64_any_dtype(df_copy[col]):
                    logger.info(f"Converting column '{col}' to datetime format.")
                    df_copy[col] = pd.to_datetime(df_copy[col], errors='coerce') # Coerce errors will turn invalid dates into NaT
                
                # Handle NaT values that might result from coercion errors
                original_null_dates = df_copy[col].isnull().sum()
                if original_null_dates > 0:
                    logger.warning(f"{original_null_dates} null values found in '{col}' after datetime conversion. Day name/weekend will be NaN for these.")

                # Add Day Name (e.g., 'Monday')
                df_copy[f'{col}_day_name'] = df_copy[col].dt.day_name()
                
                # Add Weekend Boolean (True if Saturday/Sunday)
                df_copy[f'{col}_is_weekend'] = df_copy[col].dt.dayofweek.isin([5, 6]) # 5=Sat, 6=Sun
                logger.info(f"Created '{col}_day_name' and '{col}_is_weekend' features.")
            else:
                logger.warning(f"Date column '{col}' not found in DataFrame. Skipping weekday/weekend feature creation.")
        return df_copy

    def encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies various encoding strategies to categorical features.

        Args:
            df (pd.DataFrame): The input DataFrame.

        Returns:
            pd.DataFrame: The DataFrame with encoded categorical features.
        """
        df_copy = df.copy()
        logger.info("Applying categorical encoding strategies.")

        # 1. Ordinal Encoding for FL_DATE_day_name (ordered categories)
        day_name_col = 'FL_DATE_day_name'
        if day_name_col in df_copy.columns:
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            
            # Handle potential NaNs in the column before encoding
            # OrdinalEncoder by default will treat NaN as its own category or raise error
            # handle_unknown='use_encoded_value', unknown_value=np.nan is good for new, unseen values
            # but if original data has NaNs, fill them first or handle explicitly
            
            # For OrdinalEncoder, ensure the column is of type 'category' to make sure
            # any NaNs are handled explicitly (e.g., by filling before encoding, or let encoder handle them).
            # If `handle_unknown='use_encoded_value'` and `unknown_value=np.nan` is set,
            # new categories or NaNs not in `categories` will be set to `np.nan`.
            
            # It's generally safer to fill NaNs for OrdinalEncoder
            if df_copy[day_name_col].isnull().any():
                logger.warning(f"Column '{day_name_col}' contains NaN values. OrdinalEncoder might encode them as NaN.")
                # You might choose to fill NaNs with a placeholder like 'Unknown' or mode before encoding if needed.
                # df_copy[day_name_col] = df_copy[day_name_col].fillna('Unknown') # Example
                
            ordinal_enc = OrdinalEncoder(categories=[day_order], handle_unknown='use_encoded_value', unknown_value=np.nan)
            df_copy[f'{day_name_col}_encoded'] = ordinal_enc.fit_transform(df_copy[[day_name_col]])
            logger.info(f"Applied Ordinal Encoding to '{day_name_col}'.")
        else:
            logger.warning(f"Column '{day_name_col}' not found for Ordinal Encoding.")

        # 2. One-Hot Encoding for AIRLINE (nominal)
        airline_col = 'AIRLINE'
        if airline_col in df_copy.columns:
            onehot_enc = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            airline_encoded_array = onehot_enc.fit_transform(df_copy[[airline_col]])
            airline_columns = [f'{airline_col}_{cat}' for cat in onehot_enc.categories_[0]]
            
            airline_df = pd.DataFrame(airline_encoded_array, columns=airline_columns, index=df_copy.index)
            df_copy = pd.concat([df_copy, airline_df], axis=1)
            logger.info(f"Applied One-Hot Encoding to '{airline_col}'. Created {len(airline_columns)} new columns.")
        else:
            logger.warning(f"Column '{airline_col}' not found for One-Hot Encoding.")

        # 3. Binary Encoding for high-cardinality columns
        binary_cols = ['FL_NUMBER', 'ORIGIN', 'ORIGIN_CITY', 'DEST', 'DEST_CITY', 'origin_state', 'dest_state']
        binary_cols_existing = [col for col in binary_cols if col in df_copy.columns]

        if binary_cols_existing:
            logger.info(f"Applying Binary Encoding to high-cardinality columns: {binary_cols_existing}")
            try:
                # BinaryEncoder needs numeric or string types. Coerce to string just in case.
                for col in binary_cols_existing:
                    df_copy[col] = df_copy[col].astype(str)

                binary_enc = ce.BinaryEncoder(cols=binary_cols_existing, handle_unknown='value')
                df_copy = binary_enc.fit_transform(df_copy)
                logger.info(f"Applied Binary Encoding to {len(binary_cols_existing)} columns.")
            except Exception as e:
                logger.error(f"Error applying Binary Encoding: {e}", exc_info=True)
        else:
            logger.warning("No high-cardinality columns found for Binary Encoding.")

        # --- Cleanup ---
        columns_to_drop_after_encoding = [
            'FL_DATE_day_name', 'AIRLINE',
        ]
        columns_to_drop_after_encoding_existing = [
            col for col in columns_to_drop_after_encoding if col in df_copy.columns
        ]
        if columns_to_drop_after_encoding_existing:
            df_copy = df_copy.drop(columns=columns_to_drop_after_encoding_existing)
            logger.info(f"Dropped original categorical columns after encoding: {columns_to_drop_after_encoding_existing}")
        
        return df_copy

    def create_elapsed_time_diff(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates the difference between ELAPSED_TIME and CRS_ELAPSED_TIME.

        Args:
            df (pd.DataFrame): The input DataFrame.

        Returns:
            pd.DataFrame: The DataFrame with the new 'ELAPSED_TIME_DIFF' column.
        """
        df_copy = df.copy()
        required_cols = ['ELAPSED_TIME', 'CRS_ELAPSED_TIME']
        
        if all(col in df_copy.columns for col in required_cols):
            # Ensure columns are numeric; coerce errors to NaN
            df_copy['ELAPSED_TIME'] = pd.to_numeric(df_copy['ELAPSED_TIME'], errors='coerce')
            df_copy['CRS_ELAPSED_TIME'] = pd.to_numeric(df_copy['CRS_ELAPSED_TIME'], errors='coerce')
            
            df_copy['ELAPSED_TIME_DIFF'] = df_copy['ELAPSED_TIME'] - df_copy['CRS_ELAPSED_TIME']
            logger.info("Created 'ELAPSED_TIME_DIFF' feature.")
        else:
            missing = [col for col in required_cols if col not in df_copy.columns]
            logger.warning(f"Could not create 'ELAPSED_TIME_DIFF'. Missing required columns: {missing}.")
            # Create the column with NaNs if missing to avoid downstream errors, or raise
            df_copy['ELAPSED_TIME_DIFF'] = np.nan # Ensure column exists even if cannot compute
        return df_copy


    def identify_high_correlation_pairs(self, df: pd.DataFrame, threshold: float = 0.9) -> tuple[set, list]:
        """
        Identifies highly correlated numeric columns (correlation > threshold)
        and returns a list of columns to drop to reduce multicollinearity.

        Args:
            df (pd.DataFrame): The input DataFrame.
            threshold (float): The correlation threshold (e.g., 0.9).

        Returns:
            tuple[set, list]: A tuple containing:
                - set: A set of columns identified for removal.
                - list: A list of (col1, col2, correlation_value) for highly correlated pairs.
        """
        df_numeric = df.select_dtypes(include=np.number)
        logger.info(f"Identifying highly correlated features (threshold > {threshold}).")

        # Drop columns with all NaNs before computing correlation to avoid issues
        df_numeric = df_numeric.dropna(axis=1, how='all')

        corr_matrix = df_numeric.corr().abs() # Use absolute correlation for strength
        
        # Select upper triangle of correlation matrix
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

        # Find features with correlation greater than threshold
        to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > threshold)]
        
        # Get the actual pairs for logging/inspection
        high_corr_pairs_list = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                col1 = corr_matrix.columns[i]
                col2 = corr_matrix.columns[j]
                correlation_value = corr_matrix.iloc[i, j]
                if correlation_value > threshold:
                    high_corr_pairs_list.append((col1, col2, correlation_value))
        
        logger.info(f"Found {len(high_corr_pairs_list)} highly correlated pairs (>{threshold}).")
        if high_corr_pairs_list:
            for pair in high_corr_pairs_list:
                logger.debug(f"High correlation pair: {pair[0]} vs {pair[1]} (Corr: {pair[2]:.3f})")

        # The `to_drop` list from `upper_tri` automatically selects one of the pair.
        # For example, if A-B are correlated, it will pick B to drop if B is after A.
        # If A-C are correlated, it will pick C.
        # This is a common heuristic. You might want a more sophisticated method (e.g., based on VIF, or domain knowledge)
        # but this is a good start.
        logger.info(f"Identified {len(to_drop)} columns for removal due to high correlation.")
        return set(to_drop), high_corr_pairs_list # Return set for efficient lookups, list for logging


    def exclude_columns(self, df: pd.DataFrame, columns_to_exclude: list) -> pd.DataFrame:
        """
        Excludes specified columns from the DataFrame.

        Args:
            df (pd.DataFrame): The input DataFrame.
            columns_to_exclude (list): List of column names to drop.

        Returns:
            pd.DataFrame: The DataFrame with specified columns dropped.
        """
        df_copy = df.copy()
        
        existing_columns_to_exclude = [col for col in columns_to_exclude if col in df_copy.columns]
        if existing_columns_to_exclude:
            df_copy = df_copy.drop(columns=existing_columns_to_exclude)
            logger.info(f"Excluded columns: {existing_columns_to_exclude}. Remaining columns: {len(df_copy.columns)}.")
        else:
            logger.info("No specified columns to exclude were found in the DataFrame.")
            
        return df_copy