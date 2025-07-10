# --- src/visualization.py ---
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

def plot_metric_comparison(df, output_dir="results"):
    metrics = ['MAE', 'RMSE', 'R²']
    for metric in metrics:
        plt.figure(figsize=(6, 4))
        sns.barplot(data=df, x='Model', y=metric, palette='viridis')
        plt.title(f'{metric} Comparison')
        plt.ylabel(metric)
        plt.xlabel("Model")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{metric.lower()}_comparison.png"))
        plt.close()

def plot_actual_vs_predicted(y_true, y_pred, output_path="results/actual_vs_predicted.png"):
    plt.figure(figsize=(6, 6))
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.4)
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], color='red', linestyle='--')
    plt.xlabel("Actual Arrival Delay")
    plt.ylabel("Predicted Arrival Delay")
    plt.title("Actual vs Predicted Arrival Delays")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def save_comparison_table(df, output_path="results/model_comparison.csv"):
    df.to_csv(output_path, index=False)

def write_summary(best_params, val_mae, test_mae, output_path="results/summary.txt"):
    with open(output_path, "w") as f:
        f.write("Best Neural Network Hyperparameters:\n")
        for k, v in best_params.items():
            f.write(f"- {k}: {v}\n")
        f.write(f"\nValidation MAE: {val_mae:.3f}")
        f.write(f"\nTest MAE: {test_mae:.3f}")