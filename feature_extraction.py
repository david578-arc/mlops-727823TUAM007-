import numpy as np
from scipy.signal import welch
from scipy.stats import skew, kurtosis

SAMPLING_RATE = 256
BANDS = {
    'delta': (0.5, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta':  (13, 30),
    'gamma': (30, 40),
}


def band_power(freqs, psd, low, high):
    idx = np.logical_and(freqs >= low, freqs <= high)
    return np.trapz(psd[idx], freqs[idx])


def extract_features(window):
    freqs, psd = welch(window, fs=SAMPLING_RATE, nperseg=len(window) // 2)
    features = [
        np.mean(window),
        np.std(window),
        np.var(window),
        skew(window),
        kurtosis(window),
        np.sqrt(np.mean(window ** 2)),
        np.sum(np.abs(np.diff(window))),
    ]
    for low, high in BANDS.values():
        features.append(band_power(freqs, psd, low, high))
    total_power = sum(features[-5:]) + 1e-8
    features += [p / total_power for p in features[-5:]]
    return np.array(features)


def extract_all(segments):
    return np.array([extract_features(seg) for seg in segments])
