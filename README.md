## Flight Delay Prediction Using Neural Networks ##
This machine learning project focuses on predicting arrival delays (ARR_DELAY) for commercial flights using historical flight data. Delays in air travel impact passengers, airlines, and airport operations. By accurately forecasting delays ahead of time, this model aims to support better decision-making in air traffic management, passenger notifications, and scheduling adjustments.

We developed and evaluated multiple machine learning models, and this module presents a feedforward neural network (FNN) built with TensorFlow/Keras. Our approach includes:

Feature engineering and preprocessing of flight data (e.g., carrier, departure time, weather conditions)

Use of hyperparameter optimization (HPO) to fine-tune the neural network architecture and training parameters

Application of 5-fold cross-validation (CV) to ensure robust evaluation and minimize overfitting

Performance benchmarking against a naive baseline (mean-based prediction)

The neural network significantly outperforms the baseline, achieving high predictive accuracy with low error metrics.

## File Structure ##

```
├── data/
│   ├── flights_sample_100k.csv    #raw dataset   
├── notebooks/
    ├── data-workflow.ipynb
    ├── eda_preprocessing.ipynb    #EDA, preprocessing
    ├── output_report.html         #EDA
├── results/
    ├── actual_vs_predicted.png
    ├── learning_curve.png
    ├── mae_comparison.png
    ├── model_comparison.csv
    ├── r²_comparison.png
    ├── rmse_comparison
    ├── summary.txt
├── src/
    ├── models/
        ├── baseline.py            #baseline naive mean
        ├── hpo_nn.py              #hpo
        ├── neural_net.py          #model training
    ├── evaluation.py              #evaluation of model
    ├── feature_selection.py       #top 30 features from preprocessed dataset
    ├── main.py                    #running ML pipeline
    ├── preprocessing.py           #preprocessed dataset
    ├── visualization.py           #visualization from model output
├── README.md                      #this file
├── requirements.txt
```

## Model Summary ##

The final model is a fully connected neural network with two hidden layers:

Hidden Layer 1: 128 neurons, ReLU activation

Hidden Layer 2: 64 neurons, ReLU activation

Output Layer: 1 neuron for regression (no activation)

The model is trained using the Adam optimizer and Mean Squared Error loss. EarlyStopping is applied to avoid overfitting.

Hyperparameters were optimized over a predefined grid using validation MAE as the selection criterion. The best model achieved:

MAE: 1.43

RMSE: 2.02

R²: 0.96

compared to a naive mean predictor baseline (MAE ≈ 8.50).

## Results ## 

The model’s training and validation performance was visualized using a learning curve. This helped confirm that the model generalizes well without significant overfitting. The final learning curve is saved in the results/ folder.

## How to run ##

Run the main.py file under src/ 

```
cd src
python main.py
```

#Note: make sure to install the required libraries from requirements.txt file in main branch
