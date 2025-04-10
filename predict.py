import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from data_preprocessing import load_and_preprocess_data

def predict_failure(model, new_data, sequence_length=10):
    # Prepare sequence
    X = []
    features = ['voltage', 'current', 'temperature', 'power']
    X.append(new_data[features].iloc[-sequence_length:].values)
    X = np.array(X)
    
    # Predict
    prediction = model.predict(X)[0][0]
    return prediction > 0.5, prediction  # Boolean failure, probability

if __name__ == "__main__":
    model = load_model('svc_lstm_model.h5')
    df = pd.read_csv('data/sensor_data.csv').tail(10)
    failure, prob = predict_failure(model, df)
    print(f"Predicted Failure: {failure}, Probability: {prob:.4f}")
