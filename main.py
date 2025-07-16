
# --- main.py ---
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from src.preprocessing import load_and_clean_data
from src.feature_selection import select_top_features
from src.models.hpo_nn import run_nn_kfold_with_hpo
from src.models.baseline import compare_all_baselines
from src.visualization import plot_metric_comparison, plot_actual_vs_predicted, save_comparison_table, write_summary


def main():
    df = load_and_clean_data('data/flights_sample_100k.csv')

    y = df['ARR_DELAY']
    X = df.drop(columns=['ARR_DELAY'])

    # Feature selection
    X_selected, top_features = select_top_features(X, y, top_n=30)

    # Data split
    X_temp, X_test, y_temp, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)  # 0.25 x 0.8 = 0.2

    # NN training with k-fold CV and HPO
    y_pred_val, y_pred_test, best_params, best_val_mae,history = run_nn_kfold_with_hpo(
        X_train.to_numpy(), y_train.to_numpy(),
        X_val.to_numpy(), X_test.to_numpy(),
        y_val.to_numpy(), y_test.to_numpy()
    )
 
    #Evaluate NN model
    print("Best Hyperparameters:", best_params)
    print("Validation MAE:", mean_absolute_error(y_val, y_pred_val))
    print("Test MAE:", mean_absolute_error(y_test, y_pred_test))

    # Compare baselines and NN model
    df_results = compare_all_baselines(
        X_train.to_numpy(), y_train.to_numpy(),
        X_test.to_numpy(), y_test.to_numpy(),
        y_pred_nn=y_pred_test
    )

    plot_metric_comparison(df_results)
    plot_actual_vs_predicted(y_test, y_pred_test)
    save_comparison_table(df_results)
    write_summary(best_params, mean_absolute_error(y_val, y_pred_val), mean_absolute_error(y_test, y_pred_test))
    
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Learning Curve: Train vs Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig("results/learning_curve.png", dpi=300)

if __name__ == '__main__':
    main()

 
