import pandas as pd
import numpy as np
import os
import logging
import sys # Import sys for Colab check

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
# from scripts.model_explainability import ModelExplainability # Will import this later


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

def is_running_in_colab():
    """Checks if the code is running in Google Colab."""
    return 'google.colab' in sys.modules

if __name__ == "__main__":
    is_colab = is_running_in_colab()
    logger.info(f"Running in Google Colab: {is_colab}")

    # Adjust base directory for Colab if needed
    if is_colab:
        base_dir = '/content/drive/MyDrive/Colab Notebooks/flight_delay' # A consistent base directory
        # Create the base directory if it doesn't exist
        os.makedirs(base_dir, exist_ok=True)
        # Change current working directory to the base_dir
        os.chdir(base_dir)
        logger.info(f"Changed current working directory to {os.getcwd()}")
        
        # Install kaggle if not present
        try:
            import kaggle
        except ImportError:
            logger.info("Kaggle library not found, installing...")
            os.system("pip install kaggle")
            logger.info("Kaggle library installed.")

    # Define paths relative to the current working directory
    # For Colab, this will be /content/flight_delay_prediction_project
    # For local, this will be your project root
    raw_data_dir = "data/raw"
    processed_data_dir = "data/processed"
    reports_dir = "reports"
    eda_plots_dir = "plots/eda"
    model_eval_plots_dir = "plots/model_evaluation"
    models_dir = "models"
    model_params_dir = "model_params"

    # # Ensure all necessary directories exist
    # os.makedirs(raw_data_dir, exist_ok=True)
    # os.makedirs(processed_data_dir, exist_ok=True)
    # os.makedirs(reports_dir, exist_ok=True)
    # os.makedirs(eda_plots_dir, exist_ok=True)
    # os.makedirs(model_eval_plots_dir, exist_ok=True)
    # os.makedirs(models_dir, exist_ok=True)
    # os.makedirs(model_params_dir, exist_ok=True)


    logger.info("--- Starting ML Project Pipeline ---")
    preprocessor = DataPreprocessor()
    evaluator = ModelEvaluator(output_dir=models_dir, plots_dir=model_eval_plots_dir)
    tuner = ModelTuner(output_dir=model_params_dir)
    modeler = DataModeling(tuner=tuner)

    # --- Step 1: Data Download ---
    logger.info("\n--- Step 1: Data Download ---")
    try:
        # Pass raw_data_dir to DataDownload
        downloader = DataDownload(dataset_name="patrickzel/flight-delay-and-cancellation-data-2019-2023-v2", 
                                  download_path=raw_data_dir)
        dataset_directory = downloader.data_download()
        logger.info(f"Dataset download/check complete. Data should be accessible via: {dataset_directory}")
    except Exception as e:
        logger.critical(f"Critical error during data download: {e}", exc_info=True)
        exit(1)

    # --- Step 2: Data Loading & Initial Filtering ---
    logger.info("\n--- Step 2: Data Loading & Initial Filtering ---")
    csv_file_path = os.path.join(raw_data_dir, 'flights_sample_100k.csv')
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
        report_path = os.path.join("reports", "flight_data_profile_report_pre_processing.html")
        if os.path.exists(report_path):
            logger.info(f"Profiling report already exists at: {report_path}. Skipping generation.")
        else:
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
                # This 'FLIGHT_STATUS' is for EDA/visualization, not the final modeling target.
                status = []
                for value in df_eda['ARR_DELAY']:
                    status.append(0 if value <= 10 else 1)
                df_eda['FLIGHT_STATUS_EDA'] = status # Renamed to avoid clash with modeling target
                logger.info(f"Added 'FLIGHT_STATUS_EDA' column for EDA. Value counts:\n{df_eda['FLIGHT_STATUS_EDA'].value_counts().to_string()}")
            else:
                logger.warning("'ARR_DELAY' column not found in EDA DataFrame for 'FLIGHT_STATUS_EDA' creation.")

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
            visualizer = DataVisualizer(output_dir=eda_plots_dir)

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
            # --- Convert to Classification Target ---
            if 'ARR_DELAY' in df_model.columns:
                # Create the binary target variable for classification
                df_model['FLIGHT_STATUS'] = (df_model['ARR_DELAY'] > 15).astype(int)
                logger.info(f"Created 'FLIGHT_STATUS' binary target (1 if ARR_DELAY > 15, 0 otherwise).")
                logger.info(f"FLIGHT_STATUS value counts:\n{df_model['FLIGHT_STATUS'].value_counts().to_string()}")
            else:
                logger.critical("'ARR_DELAY' column not found for creating classification target.")
                exit(1)
            
            delay_cols = ['DELAY_DUE_CARRIER', 'DELAY_DUE_WEATHER',
                          'DELAY_DUE_NAS', 'DELAY_DUE_SECURITY',
                          'DELAY_DUE_LATE_AIRCRAFT']
            df_model = preprocessor.handle_missing_values(df_model, delay_cols)
            
            # Outlier detection and treatment (for features, not target)
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
            # Exclude ARR_DELAY and FLIGHT_STATUS from correlation check for features
            df_for_corr_check = df_model.drop(columns=['ARR_DELAY', 'FLIGHT_STATUS'], errors='ignore')
            columns_to_drop_high_corr, _ = preprocessor.identify_high_correlation_pairs(df_for_corr_check, threshold=0.9)
            
            if columns_to_drop_high_corr:
                logger.info(f"Removing {len(columns_to_drop_high_corr)} columns due to high correlation: {columns_to_drop_high_corr}")
                df_model = df_model.drop(columns=list(columns_to_drop_high_corr), errors='ignore')
            else:
                logger.info("No columns identified for removal due to high correlation (threshold > 0.9).")


            # Exclude other specific columns for modeling
            # Now we drop 'ARR_DELAY' as 'FLIGHT_STATUS' is our new target
            columns_to_exclude_model = [
                'AIRLINE_DOT', 'AIRLINE_CODE', 'DOT_CODE',
                'CANCELLED', 'DIVERTED', 'CANCELLATION_CODE',
                'FLIGHT_STATUS_EDA', # Drop EDA specific column
                'CRS_DEP_TIME', 'CRS_ARR_TIME', 'WHEELS_OFF', 'WHEELS_ON', 'DEP_TIME', 'ARR_TIME',
                'FL_DATE',
                'DELAY_DUE_CARRIER', 'DELAY_DUE_WEATHER', 'DELAY_DUE_NAS',
                'DELAY_DUE_SECURITY', 'DELAY_DUE_LATE_AIRCRAFT',
                'ARR_DELAY' # This is now the source for FLIGHT_STATUS, no longer a feature
            ]
            df_model = preprocessor.exclude_columns(df_model, columns_to_exclude_model)

            logger.info("Data preprocessing for modeling complete.")
            logger.info(f"Final Modeling DataFrame shape: {df_model.shape}")
            logger.info(f"Final Modeling DataFrame columns (first 10):\n{df_model.columns.tolist()[:10]}...")
            logger.info(f"Final Modeling DataFrame columns (last 10):\n{df_model.columns.tolist()[-10:]}")
            
            output_filepath_model = os.path.join(processed_data_dir, 'preprocessed_flight_data_for_modeling.csv')
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
        
        # Now target is 'FLIGHT_STATUS'
        if 'FLIGHT_STATUS' not in df_model.columns:
            logger.critical("Target column 'FLIGHT_STATUS' not found in the modeling DataFrame. Cannot proceed with modeling.")
            exit(1)
        
        X = df_model.drop(columns=['FLIGHT_STATUS'])
        y = df_model['FLIGHT_STATUS'] # Our new classification target

        logger.info(f"Features (X) shape: {X.shape}, Target (y) shape: {y.shape}")
        logger.info(f"Target variable distribution:\n{y.value_counts()}")

        initial_X_rows = X.shape[0]
        # Drop columns that are all NaN after one-hot encoding or other processing steps
        X = X.dropna(axis=1, how='all')
        # Drop rows with any remaining NaNs (should be minimal after preprocessing)
        rows_before_final_nan_drop = X.shape[0]
        X = X.dropna()
        y = y[X.index] # Ensure y aligns with X after dropping NaNs
        
        if X.shape[0] < rows_before_final_nan_drop:
            rows_removed_nan = rows_before_final_nan_drop - X.shape[0]
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
            scaler_name = 'feature_scaler_classification' # New name for classification scaler
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

        # --- Train-Validation-Test Split ---
        # 1. Split into training+validation set and final test set
        X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
        
        # 2. Further split training+validation into true training and validation for NN
        X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=(0.15/(1-0.15)), random_state=42, stratify=y_train_val)

        logger.info(f"Data split: Train-Val (for HPO/Training)={X_train_val.shape[0]} samples, Test (final evaluation)={X_test.shape[0]} samples")
        logger.info(f"Further split of Train-Val: Train={X_train.shape[0]} samples, Validation={X_val.shape[0]} samples")
        logger.info(f"Test Set Target Distribution:\n{y_test.value_counts()}")


        # --- Model Training/Loading and Evaluation (for Classification) ---
        models_config = {
            "baseline_model": {"func": modeler.run_baseline_model_classification, "train_args": (y_train_val, y_test), "predict_data": y_test},
            "logistic_regression_model": {"func": modeler.run_logistic_regression, "train_args": (X_train_val, y_train_val), "predict_data": X_test},
            "neural_network_classification_model": {"func": modeler.run_neural_network_classification, "train_args": (X_train, y_train, X_val, y_val), "predict_data": X_test}
        }

        trained_models = {}
        for model_name, config in models_config.items():
            logger.info(f"\n--- Processing {model_name.replace('_', ' ').title()} ---")
            
            model = evaluator.load_model(model_name) # Attempt to load
            
            if model is not None:
                logger.info(f"Loaded existing model: {model_name}")
                if model_name == "baseline_model":
                    # For baseline, predict the majority class
                    majority_class = y_train_val.mode()[0]
                    y_pred_proba = np.full(len(config["predict_data"]), majority_class) # For binary, 0 or 1
                    y_pred = y_pred_proba # y_pred and y_pred_proba are the same for baseline
                elif isinstance(model, tf.keras.Model):
                    y_pred_proba = model.predict(config["predict_data"]).flatten()
                    y_pred = (y_pred_proba > 0.5).astype(int) # Convert probabilities to class labels
                else:
                    y_pred_proba = model.predict_proba(config["predict_data"])[:, 1] # Probability of positive class
                    y_pred = model.predict(config["predict_data"])
                
                trained_models[model_name] = model # Store loaded model
            else:
                logger.info(f"Training new model: {model_name}")
                if model_name == "baseline_model":
                    y_pred_proba, _ = config["func"](*config["train_args"])
                    y_pred = y_pred_proba # y_pred and y_pred_proba are the same for baseline
                    model = None
                else:
                    model = config["func"](*config["train_args"]) # Train on X_train_val or X_train/X_val
                    if isinstance(model, tf.keras.Model):
                        y_pred_proba = model.predict(config["predict_data"]).flatten()
                        y_pred = (y_pred_proba > 0.5).astype(int)
                    else:
                        y_pred_proba = model.predict_proba(config["predict_data"])[:, 1]
                        y_pred = model.predict(config["predict_data"])
                    
                    evaluator.save_model(model, model_name)
                
                trained_models[model_name] = model

            # Evaluate and plot results for Test set (final evaluation)
            # Pass y_pred_proba for classification metrics that require probabilities (like ROC AUC)
            evaluator.evaluate_classification_model(
                model_name.replace('_', ' ').title(), 
                y_test, y_pred, y_pred_proba, stage="Test"
            )
            
            # Generate plots (skip for baseline)
            if model_name != "baseline_model":
                evaluator.plot_roc_curve(model_name.replace('_', ' ').title(), y_test, y_pred_proba, stage="Test")
                evaluator.plot_confusion_matrix(model_name.replace('_', ' ').title(), y_test, y_pred, stage="Test")

        # --- Display Final Results Table ---
        evaluator.display_results_table()

    else:
        logger.warning("\n--- Data Modeling Skipped (Modeling DataFrame is None) ---")

    logger.info("\n--- ML Project Pipeline Complete ---")