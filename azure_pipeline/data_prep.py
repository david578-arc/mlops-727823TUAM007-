# 727823TUAM007
import argparse
import os
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from scipy.signal import butter, filtfilt

print(f"Roll No: 727823TUAM007 | Timestamp: {datetime.now().isoformat()}")

SAMPLING_RATE = 256
WINDOW_SIZE   = SAMPLING_RATE * 2
OVERLAP       = 0.5


def bandpass_filter(signal, lowcut=0.5, highcut=40.0, fs=SAMPLING_RATE, order=4):
    nyq = fs / 2
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    return filtfilt(b, a, signal)


def segment_signal(signal):
    step = int(WINDOW_SIZE * (1 - OVERLAP))
    return [signal[i:i + WINDOW_SIZE] for i in range(0, len(signal) - WINDOW_SIZE + 1, step)]


def generate_synthetic_data(n=100, sig_len=256 * 10, seed=42):
    np.random.seed(seed)
    normal  = np.random.randn(n, sig_len) * 0.5
    seizure = np.random.randn(n, sig_len) * 2.0 + \
              np.sin(np.linspace(0, 20 * np.pi, sig_len)) * 3
    return np.vstack([normal, seizure]), np.array([0] * n + [1] * n)


def extract_features(window):
    from scipy.stats import skew, kurtosis
    from scipy.signal import welch
    freqs, psd = welch(window, fs=SAMPLING_RATE, nperseg=len(window) // 2)
    bands = [(0.5, 4), (4, 8), (8, 13), (13, 30), (30, 40)]
    bp = [np.trapz(psd[np.logical_and(freqs >= l, freqs <= h)],
                   freqs[np.logical_and(freqs >= l, freqs <= h)]) for l, h in bands]
    total = sum(bp) + 1e-8
    return [np.mean(window), np.std(window), np.var(window),
            skew(window), kurtosis(window),
            np.sqrt(np.mean(window ** 2)), np.sum(np.abs(np.diff(window)))] + bp + [p / total for p in bp]


def main(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    signals, labels = generate_synthetic_data()

    X, y = [], []
    for signal, label in zip(signals, labels):
        filtered   = bandpass_filter(signal)
        normalized = (filtered - np.mean(filtered)) / (np.std(filtered) + 1e-8)
        for seg in segment_signal(normalized):
            X.append(extract_features(seg))
            y.append(label)

    X_scaled = StandardScaler().fit_transform(np.array(X))
    df = pd.DataFrame(X_scaled)
    df['label'] = y
    df.to_csv(os.path.join(output_dir, "processed_data.csv"), index=False)
    print(f"Data prep complete. Shape: {df.shape} | Saved to {output_dir}/processed_data.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="data_output")
    args = parser.parse_args()
    main(args.output_dir)
