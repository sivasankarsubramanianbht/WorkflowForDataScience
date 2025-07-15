import gradio as gr
import pandas as pd
import os
import sys # <--- ADDED THIS IMPORT
import logging

# Configure Logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# Remove handlers if already present to avoid duplication in case app.py is run multiple times
if logger.handlers:
    for handler in list(logger.handlers):
        logger.removeHandler(handler)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter('%(levelname)s: %(message)s')
console_handler.setFormatter(console_formatter)
logger.addHandler(console_handler)


# Define paths (ensure these match where your main.py saves the outputs)
base_dir = '.' # Assuming app.py runs from the project root or adjust accordingly
if 'google.colab' in sys.modules: # Check if running in Colab context for paths
    base_dir = '/content/drive/MyDrive/Colab Notebooks/flight_delay'

raw_data_dir = os.path.join(base_dir, "data/raw")
processed_data_dir = os.path.join(base_dir, "data/processed")
reports_dir = os.path.join(base_dir, "reports")
eda_plots_dir = os.path.join(base_dir, "plots/eda")
model_eval_plots_dir = os.path.join(base_dir, "plots/model_evaluation")
models_dir = os.path.join(base_dir, "models")
model_params_dir = os.path.join(base_dir, "model_params")


# --- Helper functions to load and display content ---

def get_basic_data_info():
    """Loads raw data and provides basic information and target variable definition."""
    try:
        csv_file_path = os.path.join(raw_data_dir, 'flights_sample_10k.csv')
        df_raw = pd.read_csv(csv_file_path)

        # Filter non-cancelled flights as in your main pipeline
        df_filtered_cancelled = df_raw[df_raw['CANCELLED'] == 0].copy()

        info_text = "### **1.2.1 Basic Data Info & Target Variable Definition**\n\n"
        info_text += f"**Raw Data Shape:** {df_raw.shape}\n\n"
        info_text += f"**Filtered (Non-Cancelled) Data Shape:** {df_filtered_cancelled.shape}\n\n"
        info_text += "**First 5 Rows of Filtered Data:**\n"
        info_text += f"```\n{df_filtered_cancelled.head().to_string()}\n```\n\n"
        info_text += "**Target Variable (`FLIGHT_STATUS`) Definition:**\n"
        info_text += "The target variable `FLIGHT_STATUS` is a binary classification target derived from `ARR_DELAY`.\n"
        info_text += "  - `FLIGHT_STATUS = 1` if `ARR_DELAY > 15` minutes (flight is considered 'delayed').\n"
        info_text += "  - `FLIGHT_STATUS = 0` if `ARR_DELAY <= 15` minutes (flight is considered 'on-time' or 'minor delay').\n\n"

        # Simulate target distribution if not explicitly saved
        if 'ARR_DELAY' in df_filtered_cancelled.columns:
            df_filtered_cancelled['FLIGHT_STATUS'] = (df_filtered_cancelled['ARR_DELAY'] > 15).astype(int)
            info_text += "**Simulated Target Distribution (from filtered raw data):**\n"
            info_text += f"```\n{df_filtered_cancelled['FLIGHT_STATUS'].value_counts().to_string()}\n```"
        else:
            info_text += "Note: 'ARR_DELAY' column not found to simulate target distribution in raw data view."

        return info_text
    except FileNotFoundError:
        return f"Error: Data file not found at {os.path.join(raw_data_dir, 'flights_sample_100k.csv')}. Please ensure the pipeline has run successfully."
    except Exception as e:
        return f"An error occurred: {e}"

def get_data_profiling_link():
    """Provides a link to the generated data profiling report."""
    report_path = os.path.join(reports_dir, "flight_data_profile_report_pre_processing.html")
    if os.path.exists(report_path):
        return f"### **1.3 Data Profiling Link**\n\n" \
               f"A comprehensive data profiling report (generated using `pandas-profiling`) is available:\n" \
               f"- [Open Data Profile Report]({report_path})\n\n" \
               f"**Note:** If running locally, click the link to open the HTML report in your browser. If in Colab, you might need to navigate to the file path in your Google Drive and open it from there."
    else:
        return f"### **1.3 Data Profiling Link**\n\n" \
               f"Data profiling report not found at: `{report_path}`. Please ensure the pipeline has generated it."

def get_eda_plots():
    """Returns a list of paths to EDA plots."""
    plot_files = []
    # List all plots generated in your pipeline's Step 5
    plots = [
        "all_column_distributions_eda.png",
        "airline_flight_counts_eda.png",
        "top_20_destination_visits_eda.png",
        "avg_arrival_delay_by_airline_eda.png",
        "total_delays_by_year_eda.png",
        "monthly_delays_by_year_eda.png",
        "monthly_delay_trend_highlight_eda.png",
        "delay_reason_breakdown_eda.png"
    ]

    found_plots_info = "### **1.2.2 EDA Plots & Inferences**\n\n"
    found_plots_info += "Explore the visualizations below to understand key trends and patterns in the flight delay data.\n\n"

    for plot in plots:
        path = os.path.join(eda_plots_dir, plot)
        if os.path.exists(path):
            plot_files.append(path)
            found_plots_info += f"- **{plot.replace('.png', '').replace('_', ' ').title()}**\n"
            # Add a brief inference for each plot based on common expectations from flight data EDA
            if "column_distributions" in plot:
                found_plots_info += "  *Inference:* Provides an overview of feature distributions, helping to identify skewed data, outliers, or categorical balance issues.\n"
            elif "airline_flight_counts" in plot:
                found_plots_info += "  *Inference:* Shows which airlines operate the most flights, indicating their market share and potential impact on overall delays.\n"
            elif "destination_visits" in plot:
                found_plots_info += "  *Inference:* Highlights the busiest destination airports, which can be hotspots for delays due to high traffic.\n"
            elif "arrival_delay_by_airline" in plot:
                found_plots_info += "  *Inference:* Compares average arrival delays across different airlines, revealing which airlines tend to be more punctual or delayed.\n"
            elif "total_delays_by_year" in plot:
                found_plots_info += "  *Inference:* Illustrates the yearly trend of total flight delays, showing if delay incidents are increasing or decreasing over time.\n"
            elif "monthly_delays_by_year" in plot:
                found_plots_info += "  *Inference:* Breaks down delays by month and year, revealing seasonal patterns and yearly variations.\n"
            elif "monthly_delay_trend_highlight" in plot:
                found_plots_info += "  *Inference:* Focuses on specific months or periods with high delay occurrences, indicating peak travel times or anomaly periods.\n"
            elif "delay_reason_breakdown" in plot:
                found_plots_info += "  *Inference:* Identifies the primary causes of delays (e.g., carrier, weather, NAS), which is crucial for targeted interventions.\n"
            found_plots_info += "\n"
        else:
            logger.warning(f"EDA plot not found: {path}")

    if not plot_files:
        return "No EDA plots found. Please ensure the pipeline has generated them.", []

    return found_plots_info, plot_files


def get_preprocessing_summary():
    """Summarizes the data cleaning and preprocessing steps."""
    summary_text = "### **1.4 Data Cleaning and Preprocessing Summary**\n\n"
    summary_text += "The raw flight data undergoes several crucial preprocessing steps to prepare it for machine learning:\n\n"
    summary_text += "1.  **Initial Filtering:** Cancelled flights (`CANCELLED == 1`) are removed as our target is arrival delay.\n"
    summary_text += "2.  **Target Variable Creation:** A new binary target `FLIGHT_STATUS` is created from `ARR_DELAY` (1 if arrival delay > 15 mins, 0 otherwise).\n"
    summary_text += "3.  **Missing Value Imputation:** Missing values in delay reason columns (`DELAY_DUE_CARRIER`, etc.) are filled, likely with zeros, indicating no delay attributed to that reason.\n"
    summary_text += "4.  **Elapsed Time Difference:** A feature `ELAPSED_TIME_DIFF` is calculated from scheduled and actual times, providing insights into flight duration deviations.\n"
    summary_text += "5.  **Cyclical Encoding:** Time-based features like `CRS_DEP_TIME`, `CRS_ARR_TIME` are transformed using sine and cosine functions to capture their cyclical nature.\n"
    summary_text += "6.  **City/State Split:** Combined city, state information in `DEST_CITY` and `ORIGIN_CITY` is separated into distinct `_CITY` and `_STATE` columns for better granularity.\n"
    summary_text += "7.  **Weekday/Weekend Features:** `FL_DATE` is used to create binary features indicating whether a flight occurs on a `WEEKDAY` or `WEEKEND`.\n"
    summary_text += "8.  **Categorical Encoding:** Categorical features (e.g., `AIRLINE`, `ORIGIN`, `DEST`, `DEST_STATE`) are converted into numerical representations, typically using one-hot encoding or similar methods, suitable for machine learning models.\n"
    summary_text += "9.  **High Correlation Feature Removal:** Features with a high correlation (e.g., >0.9) are identified and one of the pair is removed to prevent multicollinearity and improve model stability.\n"
    summary_text += "10. **Feature Exclusion:** Various columns deemed irrelevant or redundant for modeling (e.g., original identifiers, cancellation details, raw time columns, original delay values) are dropped.\n\n"
    summary_text += "The preprocessed data is then saved as `preprocessed_flight_data_for_modeling.csv`."
    return summary_text

def get_modeling_and_evaluation_results():
    """Loads and displays model evaluation results and plots."""
    results_md = "### **2. Data Modeling**\n\n"
    results_md += "Our pipeline trains and evaluates several classification models for predicting flight delays:\n"
    results_md += "- **Baseline Model:** Predicts the majority class (no delay).\n"
    results_md += "- **Logistic Regression:** A simple linear model.\n"
    results_md += "- **Neural Network:** A more complex deep learning model.\n\n"

    results_md += "### **3. Data Evaluation and KPIs (Classification)**\n\n"
    results_md += "Models are evaluated using metrics relevant for classification tasks, including:\n"
    results_md += "- **Accuracy:** Overall correctness.\n"
    results_md += "- **Precision:** Proportion of true positive predictions that were actually positive.\n"
    results_md += "- **Recall:** Proportion of actual positives that were correctly identified.\n"
    results_md += "- **F1-Score:** Harmonic mean of precision and recall.\n"
    results_md += "- **ROC AUC:** Area Under the Receiver Operating Characteristic Curve, indicating the model's ability to distinguish between classes.\n\n"

    metrics_table_path = os.path.join(model_eval_plots_dir, "classification_model_metrics_table.csv")
    metrics_display = ""
    if os.path.exists(metrics_table_path):
        try:
            df_metrics = pd.read_csv(metrics_table_path)
            metrics_display = "**Overall Model Performance (Test Set):**\n"
            metrics_display += df_metrics.to_markdown(index=False) + "\n\n"
        except Exception as e:
            metrics_display = f"Could not load metrics table: {e}\n\n"
    else:
        metrics_display = f"Metrics table not found at: `{metrics_table_path}`. Please ensure the pipeline has generated it.\n\n"

    results_md += metrics_display

    plot_files = []
    # List all evaluation plots generated in your pipeline's Step 7
    eval_plots = [
      "Logistic_Regression_Test_ROC_Curve.png",
      "Logistic_Regression_Test_Confusion_Matrix.png",
      "Neural_Network_Classification_Test_ROC_Curve.png",
      "Neural_Network_Classification_Test_Confusion_Matrix.png"
      ]

    found_plots_info = "### **Model Evaluation Plots (Test Set):**\n\n"

    for plot in eval_plots:
        path = os.path.join(model_eval_plots_dir, plot)
        if os.path.exists(path):
            plot_files.append(path)
            found_plots_info += f"- **{plot.replace('.png', '').replace('_', ' ').title()}**\n"
            if "ROC_Curve" in plot:
                found_plots_info += "  *Inference:* Visualizes the trade-off between the true positive rate and false positive rate. A curve closer to the top-left corner indicates better performance.\n"
            elif "Confusion_Matrix" in plot:
                found_plots_info += "  *Inference:* Shows the counts of true positives, true negatives, false positives, and false negatives, providing a clear picture of classification errors.\n"
            found_plots_info += "\n"
        else:
            logger.warning(f"Model evaluation plot not found: {path}")

    if not plot_files:
        found_plots_info = "No model evaluation plots found. Please ensure the pipeline has generated them."

    return results_md, found_plots_info, plot_files

def get_shap_plots():
    """Returns a list of paths to SHAP explanation plots."""
    shap_plots_dir = os.path.join(base_dir, "plots/model_explainability") # Ensure this path is defined globally or passed
    os.makedirs(shap_plots_dir, exist_ok=True) # Ensure dir exists for the app too

    plot_files = []
    shap_plots = [
        "logistic_regression_shap_summary.png",
        "neural_network_shap_summary.png",
        # Add other SHAP plots if you generate them (e.g., dependence plots)
    ]

    found_plots_info = "### **4.1 SHAP Feature Importance Plots**\n\n"
    found_plots_info += "These plots show the contribution of each feature to the model's predictions.\n\n"

    for plot in shap_plots:
        path = os.path.join(shap_plots_dir, plot)
        if os.path.exists(path):
            plot_files.append(path)
            found_plots_info += f"- **{plot.replace('.png', '').replace('_', ' ').title()}**\n"
            found_plots_info += "  *Inference:* Each point represents an instance in the dataset. The position on the x-axis shows the SHAP value, indicating the feature's impact on the prediction. Color often indicates the feature's actual value (e.g., high or low).\n\n"
        else:
            logger.warning(f"SHAP plot not found: {path}")

    if not plot_files:
        return "No SHAP plots found. Please ensure the pipeline has generated them in `plots/model_explainability`.", []

    return found_plots_info, plot_files

def get_model_explainability_info():
    """Placeholder for Model Explainability."""
    explain_md = "### **4. Model Explainability**\n\n"
    explain_md += "Understanding why a model makes certain predictions is crucial, especially in high-stakes domains like aviation.\n"
    explain_md += "This section will be dedicated to explaining model predictions using techniques such as:\n"
    explain_md += "- **SHAP (SHapley Additive exPlanations):** To explain individual predictions and global feature importance by showing the contribution of each feature to the prediction.\n"
    explain_md += "- **LIME (Local Interpretable Model-agnostic Explanations):** To explain the predictions of any machine learning model by approximating it with a local, interpretable model.\n\n"
    explain_md += "**Current Status:** This functionality is still under development. Once implemented, you will be able to select a model and input features to see how different factors influence the prediction of a flight delay.\n\n"
    explain_md += "*(Future enhancement: Interactive SHAP/LIME plots, feature importance rankings, and example predictions with explanations.)*"
    return explain_md


# --- Gradio Interface ---

with gr.Blocks(theme=gr.themes.Soft(), title="Flight Delay Prediction ML Pipeline Dashboard") as demo:
    gr.Markdown(
        """
        # ✈️ Flight Delay Prediction ML Pipeline Dashboard
        Welcome to the interactive dashboard for the Flight Delay Prediction Machine Learning project!
        This application guides you through the end-to-end ML pipeline, from data understanding and preprocessing
        to model training, evaluation, and future explainability.
        """
    )

    with gr.Tab("1. Data Overview & EDA"):
        gr.Markdown("## Data Understanding and Exploratory Data Analysis (EDA)")
        with gr.Accordion("1.1 Use Case & Pipeline Structure", open=False):
            gr.Markdown(
                """
                ### **1.1 Use Case Definition and Process Pipeline Structure**
                **Use Case:** Predict whether a flight will be significantly delayed (arrival delay > 15 minutes) or on-time/minor delay. This is a **binary classification** problem.

                **Overall Pipeline Structure:**
                1.  **Data Download:** Retrieves raw flight data from Kaggle.
                2.  **Data Loading & Initial Filtering:** Loads data and removes cancelled flights.
                3.  **Data Profiling (Pre-preprocessing):** Generates an initial report on data quality and characteristics.
                4.  **Data Preprocessing for Visualization (EDA):** Prepares data for exploratory plots (e.g., creating temporary target for EDA).
                5.  **Data Visualization (EDA):** Generates insightful plots to understand data patterns.
                6.  **Data Preprocessing for Modeling:** Cleans, transforms, and engineers features for machine learning models.
                7.  **Data Modeling and Evaluation:** Trains and evaluates various classification models.
                8.  **Model Explainability (Future):** Provides insights into model predictions.
                """
            )

        with gr.Tab("Basic Data Info"):
            gr.Markdown("---")
            gr.Markdown("### 1.2.1 Basic Info about Data and Target Variable Definition")
            basic_info_output = gr.Markdown(get_basic_data_info())

        with gr.Tab("EDA Plots"):
            gr.Markdown("---")
            eda_info_output = gr.Markdown("Loading EDA plot information...")
            eda_plot_gallery = gr.Gallery(
                label="Exploratory Data Analysis Plots",
                columns=[4], rows=[2], object_fit="contain", height="auto"
            )
            demo.load(get_eda_plots, inputs=None, outputs=[eda_info_output, eda_plot_gallery])

        with gr.Tab("Data Preprocessing Summary"):
            gr.Markdown("---")
            gr.Markdown("### 1.3 Data Cleaning and Preprocessing for Modeling")
            preprocessing_summary_output = gr.Markdown(get_preprocessing_summary())

        with gr.Tab("Data Profiling Report"):
            gr.Markdown("---")
            gr.Markdown("### 1.4 Data Profiling Link")
            profiling_link_output = gr.Markdown(get_data_profiling_link())

    with gr.Tab("2. Model Training & Evaluation"):
        gr.Markdown("## Machine Learning Model Training and Performance Evaluation")
        gr.Markdown("---")
        model_eval_summary_output = gr.Markdown("Loading model evaluation results...")
        model_eval_plot_gallery = gr.Gallery(
            label="Model Evaluation Plots (Test Set)",
            columns=[4], rows=[2], object_fit="contain", height="auto"
        )
        demo.load(get_modeling_and_evaluation_results, inputs=None, outputs=[model_eval_summary_output, model_eval_plot_gallery])

    with gr.Tab("3. Model Explainability"):
        gr.Markdown("## Understanding Model Decisions")
        gr.Markdown("---")
        # explainability_output = gr.Markdown(get_model_explainability_info()) # REMOVE THIS LINE

        shap_info_output = gr.Markdown("Loading SHAP plot information...")
        shap_plot_gallery = gr.Gallery(
            label="SHAP Explanations",
            columns=[2], rows=[1], object_fit="contain", height="auto"
        )
        demo.load(get_shap_plots, inputs=None, outputs=[shap_info_output, shap_plot_gallery])
        # Optionally, keep the general info above the plots if desired
        gr.Markdown(get_model_explainability_info())


if __name__ == "__main__":
    # Ensure all directories exist before trying to load/save files
    os.makedirs(raw_data_dir, exist_ok=True)
    os.makedirs(processed_data_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(eda_plots_dir, exist_ok=True)
    os.makedirs(model_eval_plots_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(model_params_dir, exist_ok=True)

    # You would typically run your main pipeline first to generate outputs
    # For demonstration, we assume your main.py has already run and saved artifacts.
    logger.info("Starting Gradio App. Ensure your ML pipeline (main.py) has been executed to generate necessary reports and plots.")
    demo.launch(share=True) # Set share=True to get a public link for Colab