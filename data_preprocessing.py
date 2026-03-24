import numpy as np
from scipy.signal import butter, filtfilt

SAMPLING_RATE = 256  # Hz
WINDOW_SIZE = SAMPLING_RATE * 2  # 2-second windows
OVERLAP = 0.5


def bandpass_filter(signal, lowcut=0.5, highcut=40.0, fs=SAMPLING_RATE, order=4):
    nyq = fs / 2
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    return filtfilt(b, a, signal)


def normalize(signal):
    return (signal - np.mean(signal)) / (np.std(signal) + 1e-8)


def segment_signal(signal, window_size=WINDOW_SIZE, overlap=OVERLAP):
    step = int(window_size * (1 - overlap))
    return np.array([
        signal[i:i + window_size]
        for i in range(0, len(signal) - window_size + 1, step)
    ])


def preprocess(raw_signal):
    filtered = bandpass_filter(raw_signal)
    normalized = normalize(filtered)
    return segment_signal(normalized)
