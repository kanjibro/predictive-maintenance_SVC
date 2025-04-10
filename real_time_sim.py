import time
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from data_preprocessing import generate_realistic_data
from predict import predict_failure

def simulate_real_time(model, interval=1):
    print("Starting real-time simulation... (Ctrl+C to stop)")
    df = pd.DataFrame(columns=['timestamp', 'voltage', 'current', 'temperature', 'power', 'failure'])
    
    try:
        while True:
            # Generate a new data point
            t = pd.Timestamp.now()
            voltage = 230 + np.random.normal(0, 2) + 10 * np.sin(t.hour * np.pi / 12)
            current = 50 + 0.1 * voltage + np.random.normal(0, 1)
            temperature = 35 + 0.0001 * (voltage * current) + np.random.normal(0, 0.5)
            power = voltage * current
            failure = int((voltage < 220) or (temperature > 40) or (power > 12000) or (np.random.random() < 0.02))
            
            new_row = pd.DataFrame([[t, voltage, current, temperature, power, failure]], 
                                  columns=['timestamp', 'voltage', 'current', 'temperature', 'power', 'failure'])
            df = pd.concat([df, new_row], ignore_index=True)
            
            # Predict if we have enough data
            if len(df) >= 10:
                fail, prob = predict_failure(model, df.tail(10))
                print(f"{t}: Voltage={voltage:.1f}, Current={current:.1f}, Temp={temperature:.1f}, "
                      f"Power={power:.1f}, Failure={fail} (Prob={prob:.4f})")
            
            time.sleep(interval)
    except KeyboardInterrupt:
        print("Simulation stopped.")
        df.to_csv('data/real_time_data.csv', index=False)

if __name__ == "__main__":
    model = load_model('svc_lstm_model.h5')
    simulate_real_time(model)
