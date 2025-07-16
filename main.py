# main.py
import joblib
import pandas as pd
import sys

pipeline = joblib.load('logistic_regression_pipeline.pkl')

def predict(input_dict):
    df = pd.DataFrame([input_dict])
    prediction = pipeline.predict(df)[0]
    return prediction

if __name__ == "__main__":
    # Example input: python main.py 5.3 300 1500 ...
    input_data = {
        'CRS_DEP_TIME': float(sys.argv[1]),
        'CRS_ELAPSED_TIME': float(sys.argv[2]),
        'DEP_DELAY': float(sys.argv[3]),
        'TAXI_OUT': float(sys.argv[4]),
        'TAXI_IN': float(sys.argv[5])
    }
    result = predict(input_data)
    print(f"Prediction: {'Delayed' if result == 1 else 'On Time'}")
