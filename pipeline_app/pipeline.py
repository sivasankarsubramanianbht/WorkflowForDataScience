import pandas as pd
import numpy as np
import os
import kagglehub
import logging
import sys
import importlib
import wandb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.linear_model import LogisticRegression # Ensure this is imported for isinstance checks

# Reload custom modules to ensure the latest changes are picked up
# This is crucial in environments like Jupyter/Colab where modules might be cached
import scripts.data_loading
importlib.reload(scripts.data_loading)
import scripts.data_profiling
importlib.reload(scripts.data_profiling)
import scripts.data_visualizer
importlib.reload(scripts.data_visualizer)
import scripts.data_preprocessor
importlib.reload(scripts.data_preprocessor)
import scripts.data_modelling
importlib.reload(scripts.data_modelling)
import scripts.data_evaluate
importlib.reload(scripts.data_evaluate)
import scripts.model_tuning
importlib.reload(scripts.model_tuning)
import scripts.model_explainability
importlib.reload(scripts.model_explainability)


# Custom Scripts - ensure these are available in your 'scripts' directory
from scripts.data_loading import DataDownload
from scripts.data_profiling import DataProfiler
from scripts.data_visualizer import DataVisualizer
from scripts.data_preprocessor import DataPreprocessor
from scripts.data_modelling import DataModeling
from scripts.data_evaluate import ModelEvaluator
from scripts.model_tuning import ModelTuner
from scripts.model_explainability import ModelExplainer


# --- Configure Global Logging ---
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Clear existing handlers to prevent duplicate logs if script is run multiple times
if logger.handlers:
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

file_handler = logging.FileHandler('pipeline.log')
file_handler.setLevel(logging.INFO)

console_handler = logging.StreamHandler(sys.stdout) # Use sys.stdout for console output
console_handler.setLevel(logging.INFO)

file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)

console_formatter = logging.Formatter('%(levelname)s: %(message)s')
console_handler.setFormatter(console_formatter)

logger.addHandler(file_handler)
logger.addHandler(console_handler)


class FlightDelayPipeline:
    def _download_dataset_with_fallback(self):
        """Tries to download dataset using kagglehub, falls back to DataDownload if needed."""
        try:
            path = kagglehub.dataset_download(self.config["dataset_name"])
            print("Path to dataset files:", path)
            return path
        except Exception as e:
            logger.warning(f"kagglehub download failed: {e}. Falling back to DataDownload.")
            path = self.downloader.data_download()
            print("Path to dataset files (fallback):", path)
            return path
    """
    Orchestrates the entire machine learning pipeline for flight delay prediction,
    from data download to model explainability, with WandB integration.
    """
    def __init__(self, config: dict):
        self.config = config
        self.wandb_run = None # Will be initialized in _init_wandb

        self._setup_directories()

        # Initialize core components
        self.downloader = DataDownload(
            dataset_name=self.config["dataset_name"],
            download_path=self.raw_data_dir
        )
        self.preprocessor = DataPreprocessor()
        self.profiler = DataProfiler(output_dir=self.reports_dir)
        self.visualizer = DataVisualizer(output_dir=self.eda_plots_dir)
        self.tuner = ModelTuner(output_dir=self.model_params_dir)
        self.modeler = DataModeling(tuner=self.tuner)

        # Evaluator and Explainer depend on wandb_run, so initialize them later
        self.evaluator = None
        self.explainer = None

        logger.info("FlightDelayPipeline initialized with provided configuration.")

    def _setup_directories(self):
        """Defines and creates all necessary output directories."""
        self.raw_data_dir = "data/raw"
        self.processed_data_dir = "data/processed"
        self.reports_dir = "reports"
        self.eda_plots_dir = "plots/eda"
        self.model_eval_plots_dir = "plots/model_evaluation"
        self.models_dir = "models"
        self.model_params_dir = "model_params"
        self.explainability_plots_dir = "plots/explainability"

        for directory in [
            self.raw_data_dir, self.processed_data_dir, self.reports_dir,
            self.eda_plots_dir, self.model_eval_plots_dir, self.models_dir,
            self.model_params_dir, self.explainability_plots_dir
        ]:
            os.makedirs(directory, exist_ok=True)
        logger.info("All pipeline directories ensured to exist.")

    def _handle_colab_environment(self):
        """Adjusts the working directory and installs Kaggle if running in Colab."""
        if 'google.colab' in sys.modules:
            base_dir = '/content/drive/MyDrive/Colab Notebooks/flight_delay'
            os.makedirs(base_dir, exist_ok=True)
            os.chdir(base_dir)
            logger.info(f"Running in Google Colab. Changed current working directory to {os.getcwd()}")
            try:
                import kaggle
            except ImportError:
                logger.info("Kaggle library not found, installing...")
                os.system("pip install kaggle")
                logger.info("Kaggle library installed.")
        else:
            logger.info("Not running in Google Colab environment.")


    def _init_wandb(self):
        """Initializes Weights & Biases run and sets up dependent components."""
        try:
            self.wandb_run = wandb.init(
                project=self.config["project_name"],
                name=self.config["run_name"],
                config=self.config # Log pipeline configuration
            )
            logger.info(f"WandB run initialized: {self.wandb_run.url}")
            # Initialize evaluator and explainer now that wandb_run is available
            self.evaluator = ModelEvaluator(
                output_dir=self.models_dir,
                plots_dir=self.model_eval_plots_dir,
                wandb_run=self.wandb_run
            )
            self.explainer = ModelExplainer(
                output_dir=self.explainability_plots_dir,
                wandb_run=self.wandb_run
            )
            logger.info("ModelEvaluator and ModelExplainer initialized with WandB integration.")
        except Exception as e:
            logger.error(f"Failed to initialize WandB: {e}. Proceeding without WandB logging.", exc_info=True)
            self.wandb_run = None # Ensure it's None if init fails
            # Still initialize evaluators/explainers, but without wandb_run
            self.evaluator = ModelEvaluator(
                output_dir=self.models_dir,
                plots_dir=self.model_eval_plots_dir,
                wandb_run=None
            )
            self.explainer = ModelExplainer(
                output_dir=self.explainability_plots_dir,
                wandb_run=None
            )
            logger.warning("ModelEvaluator and ModelExplainer initialized without WandB integration due to error.")


    def run(self):
        """Executes the entire machine learning pipeline."""
        self._handle_colab_environment()
        self._init_wandb() # Initialize WandB at the start of the run
        logger.info("--- Starting ML Project Pipeline ---")

        try:
            df_raw = self._step_data_download()
            df_filtered = self._step_data_load_and_filter(df_raw)
            self._step_data_profiling(df_filtered)
            df_eda = self._step_data_preprocessing_eda(df_filtered)
            self._step_data_visualization_eda(df_eda)
            df_model = self._step_data_preprocessing_modeling(df_filtered)
            self._step_data_modeling_and_evaluation(df_model)

        except Exception as e:
            logger.critical(f"Pipeline run failed: {e}", exc_info=True)
            if self.wandb_run:
                self.wandb_run.log_code(".") # Log code on pipeline failure
                self.wandb_run.finish(exit_code=1)
            sys.exit(1) # Exit with an error code

        logger.info("\n--- ML Project Pipeline Complete ---")
        if self.wandb_run:
            self.wandb_run.log_code(".") # Log code on successful completion
            self.wandb_run.finish() # End the WandB run

    def _step_data_download(self) -> pd.DataFrame:
        """Step 1: Downloads data and loads the raw CSV."""
        logger.info("\n--- Step 1: Data Download ---")
        try:
            path = self._download_dataset_with_fallback()
            csv_file_path = os.path.join(path, self.config["data_sample_file"])
            df_raw = pd.read_csv(csv_file_path)
            logger.info(f"Raw dataset loaded. Shape: {df_raw.shape}")
            if self.wandb_run:
                artifact = wandb.Artifact(name="raw-flights-data", type="dataset")
                artifact.add_dir(path)
                self.wandb_run.log_artifact(artifact)
                logger.info("Raw data artifact logged to WandB.")
            return df_raw
        except FileNotFoundError:
            logger.critical(f"Error: The expected CSV file '{self.config['data_sample_file']}' was not found after download.", exc_info=True)
            raise
        except Exception as e:
            logger.critical(f"An error occurred during data download or initial raw loading: {e}", exc_info=True)
            raise

    def _step_data_load_and_filter(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        """Step 2: Applies initial filtering for non-cancelled flights."""
        logger.info("\n--- Step 2: Data Loading & Initial Filtering ---")
        initial_shape = df_raw.shape
        df_filtered = df_raw[df_raw['CANCELLED'] == 0].copy()
        logger.info(f"Filtered for non-cancelled flights. Original shape: {initial_shape}, Filtered shape: {df_filtered.shape}")
        return df_filtered

    def _step_data_profiling(self, df: pd.DataFrame):
        """Step 3: Generates and logs a data profiling report."""
        logger.info("\n--- Step 3: Data Profiling (Pre-preprocessing) ---")
        report_name = "flight_data_profile_report_pre_processing.html"
        report_path = os.path.join(self.reports_dir, report_name)

        if os.path.exists(report_path):
            logger.info(f"Profiling report already exists at: {report_path}. Skipping generation, logging existing.")
        else:
            try:
                self.profiler.generate_profile_report(df, report_name=report_name)
                logger.info(f"Pre-processing data profiling complete. Report saved to: {report_path}")
            except Exception as e:
                logger.error(f"Error during pre-processing data profiling: {e}", exc_info=True)
                return # Do not log artifact if generation failed

        if self.wandb_run and os.path.exists(report_path):
            artifact = wandb.Artifact(name="pre_processing_data_profile", type="report")
            artifact.add_file(report_path)
            self.wandb_run.log_artifact(artifact)
            logger.info("Profiling report artifact logged to WandB.")

    def _step_data_preprocessing_eda(self, df_filtered: pd.DataFrame) -> pd.DataFrame:
        """Step 4: Preprocesses data specifically for EDA visualizations."""
        logger.info("\n--- Step 4: Data Preprocessing for Visualization (EDA) ---")
        df_eda = df_filtered.copy()
        try:
            # Convert to EDA specific target (e.g., ARR_DELAY > 10 for broad EDA)
            if 'ARR_DELAY' in df_eda.columns:
                df_eda['FLIGHT_STATUS_EDA'] = (df_eda['ARR_DELAY'] > 10).astype(int)
                logger.info("Added 'FLIGHT_STATUS_EDA' column for EDA.")

            # Add temporal features for EDA
            if 'FL_DATE' in df_eda.columns:
                df_eda['YEAR'] = pd.to_datetime(df_eda['FL_DATE']).dt.year
                df_eda['MONTH'] = pd.to_datetime(df_eda['FL_DATE']).dt.month
                logger.info("Added 'YEAR' and 'MONTH' columns for EDA time-series plots.")

            df_eda = self.preprocessor.create_elapsed_time_diff(df_eda) # Example of using a preprocessor method
            logger.info(f"EDA preprocessing complete. EDA DataFrame shape: {df_eda.shape}")
        except Exception as e:
            logger.error(f"Error during EDA data preprocessing: {e}", exc_info=True)
            df_eda = None # Indicate failure
        return df_eda

    def _step_data_visualization_eda(self, df_eda: pd.DataFrame):
        """Step 5: Generates and logs various EDA plots."""
        if df_eda is None:
            logger.warning("\n--- Data Visualization Skipped (EDA DataFrame is None) ---")
            return

        logger.info("\n--- Step 5: Data Visualization (EDA) ---")
        try:
            # Define plots and their filenames
            plots = [
                (self.visualizer.plot_column_distribution, [df_eda, 15, 4, "all_column_distributions_eda.png"], "EDA/column_distributions"),
                (self.visualizer.plot_airline_counts, [df_eda, "airline_flight_counts_eda.png"], "EDA/airline_flight_counts"),
                (self.visualizer.plot_destination_visits, [df_eda, self.config["eda_top_n_destinations"], "top_20_destination_visits_eda.png"], "EDA/top_destination_visits"),
                (self.visualizer.plot_average_arrival_delay_by_airline, [df_eda, self.config["eda_min_flight_count_airline_delay"], "avg_arrival_delay_by_airline_eda.png"], "EDA/avg_arrival_delay_by_airline"),
                (self.visualizer.plot_total_delays_by_year, [df_eda, "total_delays_by_year_eda.png"], "EDA/total_delays_by_year"),
                (self.visualizer.plot_monthly_delays_by_year, [df_eda, "monthly_delays_by_year_eda.png"], "EDA/monthly_delays_by_year"),
                (self.visualizer.plot_monthly_trend_with_highlight, [df_eda, 'ARR_DELAY', 'Monthly Total Delays Over Time', 'Total Delays (minutes)', "monthly_delay_trend_highlight_eda.png"], "EDA/monthly_delay_trend"),
                (self.visualizer.plot_delay_reason_analysis, [df_eda, "delay_reason_breakdown_eda.png"], "EDA/delay_reason_breakdown")
            ]

            for plot_func, args, wandb_key in plots:
                filename = args[-1] # Filename is always the last argument
                plot_path = os.path.join(self.eda_plots_dir, filename)
                try:
                    plot_func(*args)
                    logger.info(f"Generated {filename}")
                    if self.wandb_run:
                        self.wandb_run.log({wandb_key: wandb.Image(plot_path)})
                        logger.info(f"Logged {filename} to WandB.")
                except Exception as p_e:
                    logger.warning(f"Failed to generate or log {filename}: {p_e}", exc_info=True)

            logger.info("Data visualization complete. Plots saved to 'plots/eda/' directory and logged to WandB.")
        except Exception as e:
            logger.error(f"Error during data visualization: {e}", exc_info=True)

    def _step_data_preprocessing_modeling(self, df_filtered: pd.DataFrame) -> pd.DataFrame:
        """Step 6: Preprocesses data for machine learning modeling."""
        if df_filtered is None:
            logger.critical("\n--- Modeling Data Preprocessing Skipped (Data not loaded) ---")
            raise ValueError("Filtered DataFrame is None, cannot proceed with modeling preprocessing.")

        logger.info("\n--- Step 6: Data Preprocessing for Modeling ---")
        df_model = df_filtered.copy()

        try:
            # Convert to Classification Target
            if 'ARR_DELAY' in df_model.columns:
                df_model[self.config["target_column"]] = (df_model['ARR_DELAY'] > self.config["arr_delay_threshold_mins"]).astype(int)
                logger.info(f"Created '{self.config['target_column']}' binary target (1 if ARR_DELAY > {self.config['arr_delay_threshold_mins']}, 0 otherwise).")
                logger.info(f"{self.config['target_column']} value counts:\n{df_model[self.config['target_column']].value_counts().to_string()}")
            else:
                logger.critical("'ARR_DELAY' column not found for creating classification target. Exiting.")
                raise ValueError("'ARR_DELAY' column missing.")

            delay_cols = ['DELAY_DUE_CARRIER', 'DELAY_DUE_WEATHER',
                          'DELAY_DUE_NAS', 'DELAY_DUE_SECURITY',
                          'DELAY_DUE_LATE_AIRCRAFT']
            df_model = self.preprocessor.handle_missing_values(df_model, delay_cols)
            logger.info("Handled missing values in delay columns.")

            df_model = self.preprocessor.create_elapsed_time_diff(df_model)
            logger.info("Created elapsed time difference feature.")

            time_columns = ['CRS_DEP_TIME', 'CRS_ARR_TIME', 'WHEELS_OFF', 'WHEELS_ON', 'DEP_TIME', 'ARR_TIME']
            df_model = self.preprocessor.apply_cyclical_encoding(df_model, time_columns)
            logger.info("Applied cyclical encoding to time columns.")

            city_state_columns = ['DEST_CITY', 'ORIGIN_CITY']
            df_model = self.preprocessor.split_city_state(df_model, city_state_columns)
            logger.info("Split city-state columns.")

            date_columns = ['FL_DATE']
            df_model = self.preprocessor.add_weekday_weekend_columns(df_model, date_columns)
            logger.info("Added weekday/weekend columns.")

            columns_to_exclude_model = [
                'AIRLINE_DOT', 'AIRLINE_CODE', 'DOT_CODE',
                'CANCELLED', 'DIVERTED', 'CANCELLATION_CODE',
                'FLIGHT_STATUS_EDA', # Drop EDA specific column
                'CRS_DEP_TIME', 'CRS_ARR_TIME', 'WHEELS_OFF', 'WHEELS_ON', 'DEP_TIME', 'ARR_TIME', # Original time columns
                'FL_DATE', 'ORIGIN', 'DEST', # Original location/date columns
                'DELAY_DUE_CARRIER', 'DELAY_DUE_WEATHER', 'DELAY_DUE_NAS', # Original delay reason columns (after handling missing)
                'DELAY_DUE_SECURITY', 'DELAY_DUE_LATE_AIRCRAFT',
                'ARR_DELAY' # Original target continuous column
            ]
            df_model = self.preprocessor.exclude_columns(df_model, columns_to_exclude_model)
            logger.info(f"Excluded specified columns. Current shape: {df_model.shape}")

            df_model = self.preprocessor.encode_categorical_features(df_model)
            logger.info(f"Encoded categorical features. Current shape: {df_model.shape}")


            df_for_corr_check = df_model.drop(columns=[self.config["target_column"]], errors='ignore')
            columns_to_drop_high_corr, _ = self.preprocessor.identify_high_correlation_pairs(df_for_corr_check, threshold=self.config["high_correlation_threshold"])

            if columns_to_drop_high_corr:
                logger.info(f"Removing {len(columns_to_drop_high_corr)} columns due to high correlation (> {self.config['high_correlation_threshold']}): {columns_to_drop_high_corr}")
                df_model = df_model.drop(columns=list(columns_to_drop_high_corr), errors='ignore')
            else:
                logger.info("No columns identified for removal due to high correlation.")

            logger.info("Data preprocessing for modeling complete.")
            logger.info(f"Final Modeling DataFrame shape: {df_model.shape}")
            logger.info(f"Final Modeling DataFrame columns (first 5):\n{df_model.columns.tolist()[:5]}...")

            output_filepath_model = os.path.join(self.processed_data_dir, 'preprocessed_flight_data_for_modeling.csv')
            df_model.to_csv(output_filepath_model, index=False)
            logger.info(f"Preprocessed data for modeling saved to: {output_filepath_model}")
            if self.wandb_run:
                artifact = wandb.Artifact(name="preprocessed-modeling-data", type="processed_data")
                artifact.add_file(output_filepath_model)
                self.wandb_run.log_artifact(artifact)
                logger.info("Preprocessed data artifact logged to WandB.")

        except Exception as e:
            logger.critical(f"Critical error during data preprocessing for modeling: {e}", exc_info=True)
            df_model = None # Indicate failure
            raise # Re-raise for pipeline termination
        return df_model

    def _step_data_modeling_and_evaluation(self, df_model: pd.DataFrame):
        """Step 7: Handles data splitting, scaling, model training, and evaluation."""
        if df_model is None or df_model.empty:
            logger.critical("\n--- Data Modeling Skipped (Modeling DataFrame is None or empty) ---")
            raise ValueError("Modeling DataFrame is None or empty, cannot proceed with modeling.")

        logger.info("\n--- Step 7: Data Modeling and Evaluation ---")

        if self.config["target_column"] not in df_model.columns:
            logger.critical(f"Target column '{self.config['target_column']}' not found in the modeling DataFrame.")
            raise ValueError("Target column missing from modeling DataFrame.")

        X = df_model.drop(columns=[self.config["target_column"]])
        y = df_model[self.config["target_column"]]

        logger.info(f"Features (X) shape: {X.shape}, Target (y) shape: {y.shape}")
        logger.info(f"Target variable distribution:\n{y.value_counts().to_string()}")

        # Final NaN drop (for any remaining NaNs after encoding or other transforms)
        rows_before_final_nan_drop = X.shape[0]
        X = X.dropna(axis=0, how='any') # Drop rows with ANY NaN in features
        y = y.loc[X.index] # Align y with X
        if X.shape[0] < rows_before_final_nan_drop:
            rows_removed_nan = rows_before_final_nan_drop - X.shape[0]
            logger.warning(f"Removed {rows_removed_nan} rows due to NaN values in features after final preprocessing.")
            logger.info(f"Updated Features (X) shape: {X.shape}, Target (y) shape: {y.shape}")

        if X.empty or y.empty:
            logger.critical("Features or target DataFrame is empty after final NaN handling. Cannot proceed with modeling.")
            raise ValueError("Empty DataFrame after final NaN handling.")

        # Identify numerical columns for scaling (excluding binary and cyclical)
        numerical_features = X.select_dtypes(include=np.number).columns.tolist()
        cols_to_scale = [
            col for col in numerical_features
            if not (X[col].dropna().isin([0, 1]).all() or col.endswith('_SIN') or col.endswith('_COS'))
        ]
        logger.info(f"Identified {len(cols_to_scale)} numerical features for scaling.")

        scaler_name = 'feature_scaler_classification'
        scaler = self.evaluator.load_model(scaler_name)

        if scaler:
            X_scaled = X.copy()
            if cols_to_scale:
                X_scaled[cols_to_scale] = scaler.transform(X_scaled[cols_to_scale])
            logger.info("Loaded and applied existing StandardScaler to numerical features.")
        else:
            scaler = StandardScaler()
            X_scaled = X.copy()
            if cols_to_scale:
                X_scaled[cols_to_scale] = scaler.fit_transform(X_scaled[cols_to_scale])
            self.evaluator.save_model(scaler, scaler_name)
            logger.info("Trained and saved new StandardScaler for numerical features.")

        # Train-Validation-Test Split on SCALED data
        X_train_val, X_test, y_train_val, y_test = train_test_split(X_scaled, y, test_size=self.config["test_size"], random_state=self.config["random_state"], stratify=y)
        # Handle cases where stratify might fail for very small classes (unlikely with 10k samples)
        if len(np.unique(y_train_val)) < 2 or len(np.unique(y_test)) < 2:
             logger.warning("Stratified split resulted in fewer than two classes in train/test set. Skipping further validation split if necessary.")
             X_train, X_val, y_train, y_val = X_train_val, None, y_train_val, None # No separate validation set
        else:
            # Ensure the validation split ratio is correctly applied to the train_val set
            # test_size_for_val = self.config["validation_split_ratio"] / (1 - self.config["test_size"])
            X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=self.config["validation_split_ratio"], random_state=self.config["random_state"], stratify=y_train_val)
            logger.info(f"Further split of Train-Val: Train={X_train.shape[0]} samples, Validation={X_val.shape[0]} samples")


        logger.info(f"Data split: Train-Val (for HPO/Training)={X_train_val.shape[0]} samples, Test (final evaluation)={X_test.shape[0]} samples")
        logger.info(f"Test Set Target Distribution:\n{y_test.value_counts().to_string()}")


        models_config = {
            "baseline_model": {
                "func": self.modeler.run_baseline_model_classification,
                "train_args": (y_train_val, y_test),
                "predict_data": y_test
            },
            "logistic_regression_model": {
                "func": self.modeler.run_logistic_regression,
                "train_args": (X_train_val, y_train_val),
                "predict_data": X_test
            },
            "neural_network_classification_model": {
                "func": self.modeler.run_neural_network_classification,
                "train_args": (X_train, y_train, X_val, y_val),
                "predict_data": X_test
            }
        }

        trained_models = {}
        for model_name, cfg in models_config.items():
            logger.info(f"\n--- Processing {model_name.replace('_', ' ').title()} ---")

            model = self.evaluator.load_model(model_name)

            if model is not None:
                logger.info(f"Loaded existing model: {model_name}")
            else:
                logger.info(f"Training new model: {model_name}")
                model = cfg["func"](*cfg["train_args"])
                self.evaluator.save_model(model, model_name)

            trained_models[model_name] = model

            # Make predictions
            y_pred_proba = None
            y_pred = None

            if model_name == "baseline_model":
                majority_class = y_train_val.mode()[0]
                y_pred_proba = np.full(len(cfg["predict_data"]), float(majority_class))
                y_pred = y_pred_proba.astype(int)
            elif isinstance(model, tf.keras.Model):
                y_pred_proba = model.predict(cfg["predict_data"], verbose=0).flatten()
                y_pred = (y_pred_proba > 0.5).astype(int)
            else: # Scikit-learn models
                if hasattr(model, 'predict_proba'):
                    y_pred_proba = model.predict_proba(cfg["predict_data"])[:, 1]
                else: # For models without predict_proba (e.g., some simple estimators)
                    logger.warning(f"Model {model_name} does not have predict_proba. Using predict for probabilities (may not be well-calibrated).")
                    y_pred_proba = model.predict(cfg["predict_data"]).astype(float)
                y_pred = model.predict(cfg["predict_data"])

            # Evaluate
            self.evaluator.evaluate_classification_model(
                model_name.replace('_', ' ').title(),
                y_test, y_pred, y_pred_proba, stage="Test"
            )

            # Plot ROC and Confusion Matrix (skip for baseline)
            if model_name != "baseline_model":
                self.evaluator.plot_roc_curve(model_name.replace('_', ' ').title(), y_test, y_pred_proba, stage="Test")
                self.evaluator.plot_confusion_matrix(model_name.replace('_', ' ').title(), y_test, y_pred, stage="Test")

            # Model Explainability (skip for baseline)
            if model_name != "baseline_model" and trained_models[model_name] is not None:
                self._step_model_explainability(model_name, trained_models[model_name], X_test)
            else:
                logger.info(f"Model explainability skipped for {model_name}.")

        self.evaluator.display_results_table()

    def _step_model_explainability(self, model_name: str, model, X_test: pd.DataFrame):
        """Step 8: Generates and logs SHAP and coefficient plots for trained models."""
        if self.explainer is None:
            logger.warning("Model explainer not initialized (likely due to WandB issue). Skipping explainability plots.")
            return
        if model is None:
            logger.warning(f"Model {model_name} is None. Cannot perform explainability.")
            return

        logger.info(f"\n--- Step 8: Model Explainability ({model_name.replace('_', ' ').title()}) ---")
        try:
            X_explain_sample = X_test.sample(n=min(self.config["shap_explain_sample_size"], X_test.shape[0]), random_state=self.config["random_state"])
            task_type = 'classification'

            # SHAP Summary Plot
            self.explainer.plot_shap_summary(
                model=model,
                X=X_explain_sample,
                filename=f"{model_name}_shap_summary_plot.png",
                task_type=task_type
            )
            logger.info(f"SHAP summary plots generated for {model_name}.")

            # SHAP Dependence Plots for a few features
            # Prioritize features based on anticipated importance or from initial SHAP summary
            potential_top_features = X_explain_sample.columns.tolist()
            features_for_dependence = []

            # Smart selection of features for dependence plots
            # You might want to get top N features dynamically after running SHAP summary once
            # For now, keeping a sensible list based on expected features from your pipeline
            if 'CRS_ELAPSED_TIME' in potential_top_features: features_for_dependence.append('CRS_ELAPSED_TIME')
            if 'DISTANCE' in potential_top_features: features_for_dependence.append('DISTANCE')
            if 'AIRLINE_WN' in potential_top_features: features_for_dependence.append('AIRLINE_WN') # Example OHE feature
            if 'ORIGIN_STATE_CA' in potential_top_features: features_for_dependence.append('ORIGIN_STATE_CA') # Example OHE feature
            if 'DAY_OF_WEEK_is_weekend' in potential_top_features: features_for_dependence.append('DAY_OF_WEEK_is_weekend')
            if 'CRS_DEP_TIME_SIN' in potential_top_features: features_for_dependence.append('CRS_DEP_TIME_SIN')
            if 'MONTH_8' in potential_top_features: features_for_dependence.append('MONTH_8') # Example month from OHE

            if features_for_dependence:
                logger.info(f"Generating SHAP dependence plots for: {features_for_dependence}")
                for feature in features_for_dependence:
                    if feature in X_explain_sample.columns: # Final check
                        self.explainer.plot_shap_dependence(
                            model=model,
                            X=X_explain_sample,
                            feature=feature,
                            filename=f"{model_name}_shap_dependence_plot_{feature}.png",
                            task_type=task_type
                        )
                    else:
                        logger.warning(f"Feature '{feature}' not found in the explainability sample. Skipping dependence plot.")
            else:
                logger.info(f"No predefined or suitable features found for SHAP dependence plots for {model_name}.")

            # Coefficient Importance for linear models
            if isinstance(model, LogisticRegression):
                self.explainer.plot_coefficient_importance(
                    model=model,
                    feature_names=X_explain_sample.columns.tolist(),
                    filename=f"{model_name}_coefficient_importance.png"
                )
                logger.info(f"Coefficient importance plot generated for {model_name}.")

        except Exception as e:
            logger.error(f"Error during model explainability for {model_name.replace('_', ' ').title()}: {e}", exc_info=True)


if __name__ == "__main__":
    # Define common configuration for WandB and pipeline
    # These parameters can be easily changed for new experiments
    pipeline_config = {
        "project_name": "flight-delay-prediction-final",
        "run_name": "full_pipeline_refactored_v1",
        "dataset_name": "patrickzel/flight-delay-and-cancellation-data-2019-2023-v2",
        "data_sample_file": "flights_sample_10k.csv",
        "target_column": "FLIGHT_STATUS",
        "arr_delay_threshold_mins": 15, # For binary classification target: delay > 15 mins
        "test_size": 0.15, # Fraction of data for final test set
        "validation_split_ratio": 0.15, # Fraction of (train+val) for validation set, e.g. 0.15 of train_val
        "random_state": 42,
        "high_correlation_threshold": 0.9, # For feature removal
        "eda_top_n_destinations": 20,
        "eda_min_flight_count_airline_delay": 500,
        "shap_explain_sample_size": 1000, # Number of samples for SHAP calculations
        "nn_epochs": 10, # Neural Network training epochs
        "nn_batch_size": 32 # Neural Network training batch size
    }

    # Create and run the pipeline
    pipeline = FlightDelayPipeline(pipeline_config)
    pipeline.run()