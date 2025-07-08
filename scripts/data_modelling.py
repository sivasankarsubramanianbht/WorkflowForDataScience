import pandas as pd
import numpy as np
import logging

# For Data Splitting and Preprocessing
from sklearn.model_selection import KFold # Keep KFold for general CV setup if needed within DataModeling

# For Regression Models
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor

# For Neural Network
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# Import ModelTuner
from scripts.model_tuning import ModelTuner


# Configure a logger for this module
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

# --- Neural Network Model Definition (Helper Function) ---
def build_nn_model(input_dim, hidden_layers=1, neurons=64, dropout_rate=0.0, learning_rate=0.001):
    """
    Builds a Sequential Keras model for regression.

    Args:
        input_dim (int): Number of input features.
        hidden_layers (int): Number of hidden layers.
        neurons (int): Number of neurons per hidden layer.
        dropout_rate (float): Dropout rate for hidden layers.
        learning_rate (float): Learning rate for the Adam optimizer.

    Returns:
        tf.keras.models.Sequential: Compiled Keras model.
    """
    model = Sequential()
    model.add(Dense(neurons, input_dim=input_dim, activation='relu'))
    for _ in range(hidden_layers - 1):
        model.add(Dense(neurons, activation='relu'))
        if dropout_rate > 0:
            model.add(Dropout(dropout_rate))
    model.add(Dense(1)) # Single output neuron for regression

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='mse',
                  metrics=['mae'])
    return model

class DataModeling:
    """
    Handles training of various regression models.
    """
    def __init__(self, tuner: ModelTuner): # Accept a ModelTuner instance
        self.tuner = tuner
        logger.info("DataModeling initialized.")

    def run_baseline_model(self, y_train: pd.Series, y_test: pd.Series) -> tuple[np.ndarray, None]:
        """
        Runs a baseline model that predicts the average of the training target.

        Args:
            y_train (pd.Series): Training target values.
            y_test (pd.Series): Test target values.

        Returns:
            tuple[np.ndarray, None]: Predicted values (no model object for baseline).
        """
        logger.info("--- Running Baseline Model (Average Prediction) ---")
        y_pred_baseline = np.full(len(y_test), y_train.mean())
        return y_pred_baseline, None # No model object to return for baseline

    def run_ridge_regression(self, X_train_val: pd.DataFrame, y_train_val: pd.Series) -> Ridge:
        """
        Trains a Ridge Regression model using hyperparameter tuning on the combined
        training and validation set.

        Args:
            X_train_val (pd.DataFrame): Training + Validation features.
            y_train_val (pd.Series): Training + Validation target.

        Returns:
            Ridge: The best trained Ridge model.
        """
        logger.info("\n--- Running Ridge Regression ---")
        ridge = Ridge(random_state=42)
        param_grid = {'alpha': [0.1, 1, 10, 100]}
        
        # Attempt to load best params first
        best_params_loaded = self.tuner.get_best_params("ridge_regression")
        
        if best_params_loaded:
            logger.info("Using loaded best parameters for Ridge Regression to train final model.")
            best_ridge = Ridge(**best_params_loaded, random_state=42)
            best_ridge.fit(X_train_val, y_train_val)
        else:
            logger.info("No saved best parameters found. Performing full hyperparameter tuning for Ridge Regression.")
            best_ridge, _, _ = self.tuner.tune_model(
                model=ridge,
                param_grid=param_grid,
                X=X_train_val, # Use X_train_val for tuning
                y=y_train_val, # Use y_train_val for tuning
                model_name="ridge_regression"
            )
        return best_ridge

    def run_random_forest_regression(self, X_train_val: pd.DataFrame, y_train_val: pd.Series) -> RandomForestRegressor:
        """
        Trains a Random Forest Regressor using hyperparameter tuning on the combined
        training and validation set.

        Args:
            X_train_val (pd.DataFrame): Training + Validation features.
            y_train_val (pd.Series): Training + Validation target.

        Returns:
            RandomForestRegressor: The best trained Random Forest model.
        """
        logger.info("\n--- Running Random Forest Regression ---")
        rf_model = RandomForestRegressor(random_state=42, n_jobs=-1)
        
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 15, None],
            'min_samples_split': [2, 5, 10]
        }
        
        # Attempt to load best params first
        best_params_loaded = self.tuner.get_best_params("random_forest_regression")

        if best_params_loaded:
            logger.info("Using loaded best parameters for Random Forest Regression to train final model.")
            best_rf = RandomForestRegressor(**best_params_loaded, random_state=42, n_jobs=-1)
            best_rf.fit(X_train_val, y_train_val)
        else:
            logger.info("No saved best parameters found. Performing full hyperparameter tuning for Random Forest Regression.")
            best_rf, _, _ = self.tuner.tune_model(
                model=rf_model,
                param_grid=param_grid,
                X=X_train_val, # Use X_train_val for tuning
                y=y_train_val, # Use y_train_val for tuning
                model_name="random_forest_regression",
                search_method='random',
                n_iter=10
            )
        return best_rf

    def run_neural_network(self, X_train: pd.DataFrame, y_train: pd.Series, 
                           X_val: pd.DataFrame, y_val: pd.Series) -> tf.keras.models.Sequential:
        """
        Trains a Neural Network model. For NN, we explicitly use train and validation
        sets for training and early stopping/validation metrics, which is a common
        practice given the nature of deep learning training.

        Args:
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training target.
            X_val (pd.DataFrame): Validation features.
            y_val (pd.Series): Validation target.

        Returns:
            tf.keras.models.Sequential: The trained Keras model.
        """
        logger.info("\n--- Running Neural Network ---")
        tf.random.set_seed(42)
        np.random.seed(42)

        nn_model_params = {
            'input_dim': X_train.shape[1],
            'hidden_layers': 2,
            'neurons': 64,
            'dropout_rate': 0.1,
            'learning_rate': 0.001
        }
        
        nn_model = build_nn_model(**nn_model_params)

        logger.info(f"Neural Network Model Summary:\n{nn_model.summary()}")

        history = nn_model.fit(X_train, y_train,
                                validation_data=(X_val, y_val),
                                epochs=50,
                                batch_size=32,
                                verbose=2)
        
        return nn_model