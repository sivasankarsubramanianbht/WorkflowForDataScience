import pandas as pd
import numpy as np
import logging
import os
import json # To save best parameters
from sklearn.model_selection import KFold, GridSearchCV, RandomizedSearchCV

# Configure a logger for this module
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

class ModelTuner:
    """
    Handles hyperparameter tuning and cross-validation for machine learning models.
    """
    def __init__(self, output_dir="model_params"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        logger.info(f"ModelTuner initialized. Best parameters will be saved to '{self.output_dir}'.")

    def _save_best_params(self, model_name: str, best_params: dict):
        """
        Saves the best parameters found by hyperparameter tuning to a JSON file.

        Args:
            model_name (str): The name of the model (e.g., 'ridge_regression').
            best_params (dict): A dictionary of the best parameters.
        """
        file_path = os.path.join(self.output_dir, f'{model_name}_best_params.json')
        try:
            with open(file_path, 'w') as f:
                json.dump(best_params, f, indent=4)
            logger.info(f"Best parameters for {model_name} saved to {file_path}")
        except Exception as e:
            logger.error(f"Error saving best parameters for {model_name}: {e}")

    def _load_best_params(self, model_name: str) -> dict:
        """
        Loads the best parameters for a given model from a JSON file.

        Args:
            model_name (str): The name of the model.

        Returns:
            dict: The loaded best parameters, or an empty dictionary if not found.
        """
        file_path = os.path.join(self.output_dir, f'{model_name}_best_params.json')
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r') as f:
                    best_params = json.load(f)
                logger.info(f"Loaded best parameters for {model_name} from {file_path}")
                return best_params
            except Exception as e:
                logger.error(f"Error loading best parameters for {model_name}: {e}")
        logger.warning(f"No saved best parameters found for {model_name}.")
        return {}

    def tune_model(self, model, param_grid: dict, X: pd.DataFrame, y: pd.Series, 
                   model_name: str, cv_splits: int = 5, search_method: str = 'grid', 
                   n_iter: int = 10, scoring: str = 'neg_mean_absolute_error') -> tuple:
        """
        Performs hyperparameter tuning and cross-validation for a given model.

        Args:
            model: The scikit-learn estimator to tune.
            param_grid (dict): Dictionary with parameters names (str) as keys and lists of parameter
                                settings to try as values.
            X (pd.DataFrame): Training features.
            y (pd.Series): Training target.
            model_name (str): A descriptive name for the model (e.g., 'ridge_regression').
            cv_splits (int): Number of cross-validation splits.
            search_method (str): 'grid' for GridSearchCV or 'random' for RandomizedSearchCV.
            n_iter (int): Number of parameter settings that are sampled if search_method is 'random'.
            scoring (str): Scoring metric to optimize (e.g., 'neg_mean_absolute_error').

        Returns:
            tuple: A tuple containing (best_estimator, best_params, best_score).
        """
        logger.info(f"Starting hyperparameter tuning for {model_name} using {search_method} search...")
        
        cv_strategy = KFold(n_splits=cv_splits, shuffle=True, random_state=42)

        if search_method == 'grid':
            search = GridSearchCV(model, param_grid, cv=cv_strategy, scoring=scoring, verbose=1, n_jobs=-1)
        elif search_method == 'random':
            search = RandomizedSearchCV(model, param_grid, n_iter=n_iter, cv=cv_strategy, scoring=scoring, verbose=1, n_jobs=-1, random_state=42)
        else:
            raise ValueError(f"Unknown search_method: {search_method}. Must be 'grid' or 'random'.")

        search.fit(X, y)

        best_estimator = search.best_estimator_
        best_params = search.best_params_
        best_score = search.best_score_

        logger.info(f"Best parameters for {model_name}: {best_params}")
        logger.info(f"Best CV score for {model_name} ({scoring}): {best_score:.4f}")

        self._save_best_params(model_name, best_params)
        
        return best_estimator, best_params, best_score

    def get_best_params(self, model_name: str) -> dict:
        """
        Retrieves the best parameters for a model, loading them if already saved.

        Args:
            model_name (str): The name of the model.

        Returns:
            dict: The best parameters dictionary.
        """
        return self._load_best_params(model_name)