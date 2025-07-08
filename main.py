import pandas as pd
import numpy as np
import os
import logging

# For Data Splitting and Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Custom Scripts
from scripts.data_loading import DataDownload
from scripts.data_profiling import DataProfiler
from scripts.data_visualizer import DataVisualizer
from scripts.data_preprocessor import DataPreprocessor
from scripts.data_modelling import DataModeling
from scripts.model_evaluate import ModelEvaluator
from scripts.model_tuning import ModelTuner

# --- Configure Logging ---
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

file_handler = logging.FileHandler('pipeline.log')
file_handler.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)

console_formatter = logging.Formatter('%(levelname)s: %(message)s')
console_handler.setFormatter(console_formatter)

if not logger.handlers:
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

if __name__ == "__main__":
    logger.info("--- Starting ML Project Pipeline ---")
    preprocessor = DataPreprocessor()
    evaluator = ModelEvaluator()
    tuner = ModelTuner()
    modeler = DataModeling(tuner=tuner) # Pass tuner instance to DataModeling

    # --- Step 1: Data Download ---
    logger.info("\n--- Step 1: Data Download ---")
    try:
        downloader = DataDownload(dataset_name="patrickzel/flight-delay-and-cancellation-data-2019-2023-v2")
        dataset_directory = downloader.data_download()
        logger.info(f"Dataset download/check complete. Data should be accessible via: {dataset_directory}")
    except Exception as e:
        logger.critical(f"Critical error during data download: {e}", exc_info=True)
        exit(1)

    # --- Step 2: Data Loading & Initial Filtering ---
    logger.info("\n--- Step 2: Data Loading & Initial Filtering ---")
    csv_file_path = os.path.join(dataset_directory, 'flights_sample_100k.csv')
    df_raw = None

    logger.info(f"Attempting to load CSV from: {csv_file_path}")
    try:
        df_raw = pd.read_csv(csv_file_path)
        logger.info("Dataset loaded successfully!")
        logger.info(f"First 5 rows of the raw dataframe:\n{df_raw.head().to_string()}")
        logger.info(f"Shape of the raw dataframe: {df_raw.shape}")
        
        initial_shape_before_filter = df_raw.shape
        df_filtered_cancelled = df_raw[df_raw['CANCELLED'] == 0].copy()
        logger.info(f"Filtered for non-cancelled flights. Original shape: {initial_shape_before_filter}, Filtered shape: {df_filtered_cancelled.shape}")

    except FileNotFoundError:
        logger.critical(f"Error: The expected CSV file '{csv_file_path}' was not found. Cannot proceed.", exc_info=True)
        try:
            parent_dir = os.path.dirname(csv_file_path)
            if os.path.exists(parent_dir):
                logger.critical(f"Contents of directory '{parent_dir}': {os.listdir(parent_dir)}")
            else:
                logger.critical(f"Parent directory '{parent_dir}' does not exist.")
        except Exception as e_os:
            logger.critical(f"Could not list directory contents for debugging: {e_os}")
        exit(1)
    except Exception as e:
        logger.critical(f"An unexpected error occurred during data loading: {e}", exc_info=True)
        exit(1)

    # --- Step 3: Data Profiling (Pre-preprocessing) ---
    if df_filtered_cancelled is not None:
        logger.info("\n--- Step 3: Data Profiling (Pre-preprocessing) ---")
        try:
            profiler = DataProfiler(output_dir="reports")
            report_path = profiler.generate_profile_report(df_filtered_cancelled, report_name="flight_data_profile_report_pre_processing.html")
            logger.info(f"Pre-processing data profiling complete. Report saved to: {report_path}")
        except Exception as e:
            logger.error(f"Error during pre-processing data profiling: {e}", exc_info=True)
    else:
        logger.warning("\n--- Data Profiling Skipped (DataFrame is None) ---")

    # --- Step 4: Data Preprocessing for Visualization (EDA) ---
    df_eda = None
    if df_filtered_cancelled is not None:
        logger.info("\n--- Step 4: Data Preprocessing for Visualization (EDA) ---")
        df_eda = df_filtered_cancelled.copy()

        try:
            if 'ARR_DELAY' in df_eda.columns:
                status = []
                for value in df_eda['ARR_DELAY']:
                    status.append(0 if value <= 10 else 1)
                df_eda['FLIGHT_STATUS'] = status
                logger.info(f"Added 'FLIGHT_STATUS' column for EDA. Value counts:\n{df_eda['FLIGHT_STATUS'].value_counts().to_string()}")
            else:
                logger.warning("'ARR_DELAY' column not found in EDA DataFrame for 'FLIGHT_STATUS' creation.")

            if 'FL_DATE' in df_eda.columns:
                df_eda['YEAR'] = pd.to_datetime(df_eda['FL_DATE']).dt.year
                df_eda['MONTH'] = pd.to_datetime(df_eda['FL_DATE']).dt.month
                logger.info("Added 'YEAR' and 'MONTH' columns for EDA time-series plots.")

            df_eda = preprocessor.create_elapsed_time_diff(df_eda)

            logger.info(f"EDA preprocessing complete. EDA DataFrame shape: {df_eda.shape}")
            
        except Exception as e:
            logger.error(f"Error during EDA data preprocessing: {e}", exc_info=True)
            df_eda = None

    else:
        logger.warning("\n--- EDA Data Preprocessing Skipped (Data not loaded) ---")


    # --- Step 5: Data Visualization (EDA) ---
    if df_eda is not None:
        logger.info("\n--- Step 5: Data Visualization (EDA) ---")
        try:
            visualizer = DataVisualizer(output_dir="plots/eda")

            visualizer.plot_column_distribution(df_eda, n_graph_shown=15, n_graph_per_row=4, filename="all_column_distributions_eda.png")
            visualizer.plot_airline_counts(df_eda, filename="airline_flight_counts_eda.png")
            visualizer.plot_destination_visits(df_eda, top_n=20, filename="top_20_destination_visits_eda.png")
            visualizer.plot_average_arrival_delay_by_airline(df_eda, min_flight_count=500, filename="avg_arrival_delay_by_airline_eda.png")
            visualizer.plot_total_delays_by_year(df_eda, filename="total_delays_by_year_eda.png")
            visualizer.plot_monthly_delays_by_year(df_eda, filename="monthly_delays_by_year_eda.png")
            visualizer.plot_monthly_trend_with_highlight(df_eda, 'ARR_DELAY', 'Monthly Total Delays Over Time', 'Total Delays (minutes)', filename="monthly_delay_trend_highlight_eda.png")
            visualizer.plot_delay_reason_analysis(df_eda, filename="delay_reason_breakdown_eda.png")
            
            logger.info("Data visualization complete. Plots saved to 'plots/eda/' directory.")
        except Exception as e:
            logger.error(f"Error during data visualization: {e}", exc_info=True)
    else:
        logger.warning("\n--- Data Visualization Skipped (EDA DataFrame is None) ---")


    # --- Step 6: Data Preprocessing for Modeling ---
    df_model = None
    if df_filtered_cancelled is not None:
        logger.info("\n--- Step 6: Data Preprocessing for Modeling ---")
        df_model = df_filtered_cancelled.copy()

        try:
            delay_cols = ['DELAY_DUE_CARRIER', 'DELAY_DUE_WEATHER',
                          'DELAY_DUE_NAS', 'DELAY_DUE_SECURITY',
                          'DELAY_DUE_LATE_AIRCRAFT']
            df_model = preprocessor.handle_missing_values(df_model, delay_cols)
            
            # Outlier detection and treatment
            df_model = preprocessor.remove_outliers_iqr(df_model)

            # Create ELAPSED_TIME_DIFF feature
            df_model = preprocessor.create_elapsed_time_diff(df_model)

            # Cyclical encoding
            time_columns = ['CRS_DEP_TIME', 'CRS_ARR_TIME', 'WHEELS_OFF', 'WHEELS_ON', 'DEP_TIME', 'ARR_TIME']
            df_model = preprocessor.apply_cyclical_encoding(df_model, time_columns)

            # Split city and state
            city_state_columns = ['DEST_CITY', 'ORIGIN_CITY']
            df_model = preprocessor.split_city_state(df_model, city_state_columns)

            # Add weekday/weekend columns
            date_columns = ['FL_DATE']
            df_model = preprocessor.add_weekday_weekend_columns(df_model, date_columns)

            # Encoding for Categorical Data
            df_model = preprocessor.encode_categorical_features(df_model)
            
            # --- High Correlation Feature Removal ---
            df_for_corr_check = df_model.drop(columns=['ARR_DELAY'], errors='ignore')
            columns_to_drop_high_corr, _ = preprocessor.identify_high_correlation_pairs(df_for_corr_check, threshold=0.9)
            
            if columns_to_drop_high_corr:
                logger.info(f"Removing {len(columns_to_drop_high_corr)} columns due to high correlation: {columns_to_drop_high_corr}")
                df_model = df_model.drop(columns=list(columns_to_drop_high_corr), errors='ignore')
            else:
                logger.info("No columns identified for removal due to high correlation (threshold > 0.9).")


            # Exclude other specific columns for modeling
            columns_to_exclude_model = [
                'AIRLINE_DOT', 'AIRLINE_CODE', 'DOT_CODE',
                'CANCELLED', 'DIVERTED', 'CANCELLATION_CODE',
                'FLIGHT_STATUS',
                'CRS_DEP_TIME', 'CRS_ARR_TIME', 'WHEELS_OFF', 'WHEELS_ON', 'DEP_TIME', 'ARR_TIME',
                'FL_DATE',
                'DELAY_DUE_CARRIER', 'DELAY_DUE_WEATHER', 'DELAY_DUE_NAS',
                'DELAY_DUE_SECURITY', 'DELAY_DUE_LATE_AIRCRAFT',
            ]
            df_model = preprocessor.exclude_columns(df_model, columns_to_exclude_model)

            logger.info("Data preprocessing for modeling complete.")
            logger.info(f"Final Modeling DataFrame shape: {df_model.shape}")
            logger.info(f"Final Modeling DataFrame columns (first 10):\n{df_model.columns.tolist()[:10]}...")
            logger.info(f"Final Modeling DataFrame columns (last 10):\n{df_model.columns.tolist()[-10:]}")
            
            output_data_dir = "data/processed"
            os.makedirs(output_data_dir, exist_ok=True)
            output_filepath_model = os.path.join(output_data_dir, 'preprocessed_flight_data_for_modeling.csv')
            df_model.to_csv(output_filepath_model, index=False)
            logger.info(f"Preprocessed data for modeling saved to: {output_filepath_model}")

        except Exception as e:
            logger.critical(f"Critical error during data preprocessing for modeling: {e}", exc_info=True)
            df_model = None
            exit(1)
    else:
        logger.warning("\n--- Modeling Data Preprocessing Skipped (Data not loaded) ---")

    # --- Step 7: Data Modeling and Evaluation ---
    if df_model is not None:
        logger.info("\n--- Step 7: Data Modeling and Evaluation ---")
        
        if 'ARR_DELAY' not in df_model.columns:
            logger.critical("Target column 'ARR_DELAY' not found in the modeling DataFrame. Cannot proceed with modeling.")
            exit(1)
        
        X = df_model.drop(columns=['ARR_DELAY'])
        y = df_model['ARR_DELAY']

        logger.info(f"Features (X) shape: {X.shape}, Target (y) shape: {y.shape}")
        
        initial_X_rows = X.shape[0]
        X = X.dropna(axis=1, how='all')
        X = X.dropna()
        y = y[X.index] # Ensure y aligns with X after dropping NaNs
        
        if X.shape[0] < initial_X_rows:
            rows_removed_nan = initial_X_rows - X.shape[0]
            logger.warning(f"Removed {rows_removed_nan} rows due to NaN values in features after final preprocessing.")
            logger.info(f"Updated Features (X) shape: {X.shape}, Target (y) shape: {y.shape}")

        if X.empty or y.empty:
            logger.critical("Features or target DataFrame is empty after NaN handling. Cannot proceed with modeling.")
            exit(1)

        # Scale numerical features - this scaler must be saved/loaded as well if used for new predictions
        numerical_features = X.select_dtypes(include=np.number).columns.tolist()
        
        cols_to_scale = []
        for col in numerical_features:
            is_binary = X[col].dropna().isin([0, 1]).all()
            if not is_binary and not (col.endswith('_SIN') or col.endswith('_COS')):
                cols_to_scale.append(col)
        
        if cols_to_scale:
            scaler_name = 'feature_scaler'
            scaler = evaluator.load_model(scaler_name) # Try loading the scaler
            
            if scaler:
                X[cols_to_scale] = scaler.transform(X[cols_to_scale])
                logger.info("Loaded and applied existing StandardScaler to numerical features.")
            else:
                scaler = StandardScaler()
                X[cols_to_scale] = scaler.fit_transform(X[cols_to_scale])
                evaluator.save_model(scaler, scaler_name)
                logger.info("Trained and saved new StandardScaler for numerical features.")
        else:
            logger.info("No numerical features identified for scaling or already scaled/binary.")

        # --- Train-Validation-Test Split (Revised) ---
        # 1. Split into training+validation set and final test set
        X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
        
        # 2. Further split training+validation into true training and validation for NN (or if other models need explicit val set)
        # 15% of 85% is ~13% of total data for X_val
        X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=(0.15/(1-0.15)), random_state=42) # Adjust size ratio

        logger.info(f"Data split: Train-Val (for HPO/Training)={X_train_val.shape[0]} samples, Test (final evaluation)={X_test.shape[0]} samples")
        logger.info(f"Further split of Train-Val: Train={X_train.shape[0]} samples, Validation={X_val.shape[0]} samples")


        # --- Model Training/Loading and Evaluation ---
        models_config = {
            "baseline_model": {"func": modeler.run_baseline_model, "train_args": (y_train_val, y_test), "predict_data": y_test}, # Note: predict_data is y_test for baseline
            "ridge_regression_model": {"func": modeler.run_ridge_regression, "train_args": (X_train_val, y_train_val), "predict_data": X_test},
            "random_forest_model": {"func": modeler.run_random_forest_regression, "train_args": (X_train_val, y_train_val), "predict_data": X_test},
            "neural_network_model": {"func": modeler.run_neural_network, "train_args": (X_train, y_train, X_val, y_val), "predict_data": X_test}
        }

        trained_models = {}
        for model_name, config in models_config.items():
            logger.info(f"\n--- Processing {model_name.replace('_', ' ').title()} ---")
            
            model = evaluator.load_model(model_name) # Attempt to load
            
            if model is not None:
                logger.info(f"Loaded existing model: {model_name}")
                # Baseline model does not have a 'predict' method or model object
                if model_name == "baseline_model":
                    # Recalculate baseline prediction for consistent evaluation, no model object
                    y_pred = np.full(len(config["predict_data"]), config["train_args"][0].mean()) # Use y_train_val mean
                elif isinstance(model, tf.keras.Model):
                    y_pred = model.predict(config["predict_data"]).flatten()
                else:
                    y_pred = model.predict(config["predict_data"])
                
                trained_models[model_name] = model # Store loaded model
            else:
                logger.info(f"Training new model: {model_name}")
                # Call the modeler's training function
                if model_name == "baseline_model":
                    y_pred, _ = config["func"](*config["train_args"]) # Baseline returns y_pred directly
                    model = None # No model object to save
                else:
                    model = config["func"](*config["train_args"]) # Train on X_train_val or X_train/X_val
                    if isinstance(model, tf.keras.Model):
                        y_pred = model.predict(config["predict_data"]).flatten()
                    else:
                        y_pred = model.predict(config["predict_data"])
                    
                    evaluator.save_model(model, model_name) # Save the newly trained model
                
                trained_models[model_name] = model # Store newly trained model

            # Evaluate and plot results for Test set (final evaluation)
            evaluator.evaluate_regression_model(model_name.replace('_', ' ').title(), y_test, y_pred, stage="Test")
            
            # Generate plots (skip for baseline)
            if model_name != "baseline_model":
                evaluator.plot_predictions_vs_actual(model_name.replace('_', ' ').title(), y_test, y_pred, stage="Test")
                evaluator.plot_residuals(model_name.replace('_', ' ').title(), y_test, y_pred, stage="Test")

        # --- Display Final Results Table ---
        evaluator.display_results_table()

    else:
        logger.warning("\n--- Data Modeling Skipped (Modeling DataFrame is None) ---")

    logger.info("\n--- ML Project Pipeline Complete ---")