import pandas as pd
import matplotlib.pyplot as plt
from data_preprocessing import load_and_preprocess_data
from model_training import train_model

def plot_sensor_trends(df):
    plt.figure(figsize=(12, 6))
    plt.plot(df['timestamp'], df['voltage'], label='Voltage')
    plt.plot(df['timestamp'], df['current'], label='Current')
    plt.plot(df['timestamp'], df['temperature'], label='Temperature')
    plt.legend()
    plt.title('SVC Sensor Trends')
    plt.xlabel('Timestamp')
    plt.ylabel('Value')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('sensor_trends.png')
    plt.close()

def plot_training_history(history):
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('training_history.png')
    plt.close()

def plot_failure_predictions(df, model):
    X, y = load_and_preprocess_data(sequence_length=10)
    preds = model.predict(X)
    plt.figure(figsize=(12, 6))
    plt.plot(df['timestamp'][10:], y, label='Actual Failures')
    plt.plot(df['timestamp'][10:], preds, label='Predicted Probabilities')
    plt.title('Failure Predictions vs Actual')
    plt.xlabel('Timestamp')
    plt.ylabel('Probability/Failure')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('failure_predictions.png')
    plt.close()

if __name__ == "__main__":
    df = pd.read_csv('data/sensor_data.csv')
    model, history = train_model()
    plot_sensor_trends(df)
    plot_training_history(history)
    plot_failure_predictions(df, model)
    print("Visualizations saved as PNG files.")
