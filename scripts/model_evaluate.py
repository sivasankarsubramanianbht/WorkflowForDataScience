import pandas as pd
import numpy as np
import logging
import os
import joblib # For saving/loading scikit-learn models
import tensorflow as tf # For saving/loading Keras models
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Configure a logger for this module
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

class ModelEvaluator:
    """
    Handles evaluation, reporting, and saving/loading of trained machine learning models.
    """
    def __init__(self, output_dir="models", plots_dir="plots/model_evaluation"):
        self.output_dir = output_dir
        self.plots_dir = plots_dir
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)
        self.results_table = [] # To store metrics for all runs
        logger.info(f"ModelEvaluator initialized. Models will be saved to '{self.output_dir}'.")
        logger.info(f"Evaluation plots will be saved to '{self.plots_dir}'.")

    def evaluate_regression_model(self, model_name: str, y_true: pd.Series, y_pred: np.ndarray, 
                                  stage: str = "Test") -> dict:
        """
        Calculates and returns common regression metrics.

        Args:
            model_name (str): Name of the model being evaluated.
            y_true (pd.Series): True target values.
            y_pred (np.ndarray): Predicted target values.
            stage (str): The stage of evaluation (e.g., "Train-Val CV", "Test").

        Returns:
            dict: A dictionary containing MAE, RMSE, and R2 scores.
        """
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        
        logger.info(f"Evaluation for {model_name} on {stage} Set: MAE={mae:.4f}, RMSE={rmse:.4f}, R2={r2:.4f}")
        
        metrics = {'Model': model_name, 'Stage': stage, 'MAE': mae, 'RMSE': rmse, 'R2': r2}
        self.results_table.append(metrics) # Store results for summary table
        return metrics

    def plot_predictions_vs_actual(self, model_name: str, y_true: pd.Series, y_pred: np.ndarray, stage: str = "Test"):
        """
        Generates a scatter plot of predicted vs. actual values.

        Args:
            model_name (str): Name of the model.
            y_true (pd.Series): True target values.
            y_pred (np.ndarray): Predicted target values.
            stage (str): The stage of evaluation (e.g., "Train-Val CV", "Test").
        """
        try:
            plt.figure(figsize=(10, 7))
            sns.scatterplot(x=y_true, y=y_pred, alpha=0.3)
            # Add a perfect prediction line
            min_val = min(y_true.min(), y_pred.min())
            max_val = max(y_true.max(), y_pred.max())
            plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
            
            plt.title(f'{model_name}: Predicted vs. Actual Delay ({stage} Set)')
            plt.xlabel('Actual Delay (minutes)')
            plt.ylabel('Predicted Delay (minutes)')
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.legend()
            plot_name = f'{model_name.replace(" ", "_").lower()}_predictions_vs_actual_{stage.lower().replace(" ", "_")}.png'
            plot_path = os.path.join(self.plots_dir, plot_name)
            plt.savefig(plot_path)
            plt.close()
            logger.info(f"Saved predicted vs. actual plot for {model_name} ({stage} Set) to {plot_path}")
        except Exception as e:
            logger.error(f"Error generating predictions vs. actual plot for {model_name}: {e}")

    def plot_residuals(self, model_name: str, y_true: pd.Series, y_pred: np.ndarray, stage: str = "Test"):
        """
        Generates a scatter plot of residuals (actual - predicted) vs. predicted values.

        Args:
            model_name (str): Name of the model.
            y_true (pd.Series): True target values.
            y_pred (np.ndarray): Predicted target values.
            stage (str): The stage of evaluation (e.g., "Train-Val CV", "Test").
        """
        try:
            residuals = y_true - y_pred
            plt.figure(figsize=(10, 7))
            sns.scatterplot(x=y_pred, y=residuals, alpha=0.3)
            plt.axhline(y=0, color='r', linestyle='--', lw=2)
            plt.title(f'{model_name}: Residuals Plot ({stage} Set)')
            plt.xlabel('Predicted Delay (minutes)')
            plt.ylabel('Residuals (Actual - Predicted)')
            plt.grid(True, linestyle='--', alpha=0.6)
            plot_name = f'{model_name.replace(" ", "_").lower()}_residuals_plot_{stage.lower().replace(" ", "_")}.png'
            plot_path = os.path.join(self.plots_dir, plot_name)
            plt.savefig(plot_path)
            plt.close()
            logger.info(f"Saved residuals plot for {model_name} ({stage} Set) to {plot_path}")
        except Exception as e:
            logger.error(f"Error generating residuals plot for {model_name}: {e}")

    def save_model(self, model, model_name: str):
        """
        Saves a trained model to the specified output directory.

        Args:
            model: The trained model object (scikit-learn or Keras).
            model_name (str): A descriptive name for the model (e.g., 'ridge_regression_model').
        """
        try:
            if isinstance(model, tf.keras.Model):
                # Keras models
                model_path = os.path.join(self.output_dir, f'{model_name}.h5') # Keras recommended format
                model.save(model_path)
                logger.info(f"Keras model '{model_name}' saved successfully to {model_path}")
            else:
                # Scikit-learn models
                model_path = os.path.join(self.output_dir, f'{model_name}.joblib')
                joblib.dump(model, model_path)
                logger.info(f"Scikit-learn model '{model_name}' saved successfully to {model_path}")
        except Exception as e:
            logger.error(f"Error saving model '{model_name}': {e}", exc_info=True)

    def load_model(self, model_name: str):
        """
        Loads a trained model from the specified output directory.

        Args:
            model_name (str): The name of the model to load (without extension).

        Returns:
            The loaded model object, or None if loading fails.
        """
        # Try loading as Keras model first
        keras_path = os.path.join(self.output_dir, f'{model_name}.h5')
        if os.path.exists(keras_path):
            try:
                model = tf.keras.models.load_model(keras_path)
                logger.info(f"Keras model '{model_name}' loaded successfully from {keras_path}")
                return model
            except Exception as e:
                logger.error(f"Error loading Keras model '{model_name}' from {keras_path}: {e}", exc_info=True)
                # Fallback to joblib if it was saved incorrectly or with a different format
        
        # Then try loading as scikit-learn model
        sklearn_path = os.path.join(self.output_dir, f'{model_name}.joblib')
        if os.path.exists(sklearn_path):
            try:
                model = joblib.load(sklearn_path)
                logger.info(f"Scikit-learn model '{model_name}' loaded successfully from {sklearn_path}")
                return model
            except Exception as e:
                logger.error(f"Error loading scikit-learn model '{model_name}' from {sklearn_path}: {e}", exc_info=True)
        
        logger.warning(f"Model '{model_name}' not found at {keras_path} or {sklearn_path}.")
        return None

    def display_results_table(self):
        """
        Displays all collected model evaluation results in a formatted table.
        """
        if not self.results_table:
            logger.info("No model evaluation results to display.")
            return

        df_results = pd.DataFrame(self.results_table)
        logger.info("\n--- Model Evaluation Summary ---")
        logger.info(f"\n{df_results.to_string(index=False)}")
        
        # Optionally save the table to a CSV or markdown file
        table_path = os.path.join(self.plots_dir, 'model_evaluation_summary.csv')
        df_results.to_csv(table_path, index=False)
        logger.info(f"Model evaluation summary table saved to {table_path}")