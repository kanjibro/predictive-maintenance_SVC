import pandas as pd
import numpy as np

def generate_csv(n_samples=1000, output_file='data/sensor_data.csv'):
    time = pd.date_range(start='2025-01-01', periods=n_samples, freq='1min')
    voltage = 230 + 10 * np.sin(np.linspace(0, 2 * np.pi, n_samples)) + np.random.normal(0, 2, n_samples)
    current = 50 + 0.1 * voltage + np.random.normal(0, 1, n_samples)
    temperature = 35 + 0.0001 * (voltage * current) + np.sin(np.linspace(0, np.pi, n_samples)) + np.random.normal(0, 0.5, n_samples)
    power = voltage * current
    failure = ((voltage < 220) | (temperature > 40) | (power > 12000) | (np.random.random(n_samples) < 0.02)).astype(int)

    df = pd.DataFrame({
        'timestamp': time,
        'voltage': voltage,
        'current': current,
        'temperature': temperature,
        'power': power,
        'failure': failure
    })
    df.to_csv(output_file, index=False)
    print(f"Generated {output_file} with {n_samples} rows.")

if __name__ == "__main__":
    generate_csv()
