import os
import sys
import warnings
import logging
import numpy as np
import mlflow
import mlflow.sklearn

warnings.filterwarnings("ignore")
logging.getLogger("mlflow").setLevel(logging.ERROR)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from models import random_forest as rf_model
from models import gradient_boosting as gb_model
from data_preprocessing import preprocess
from feature_extraction import extract_all

ROLL_NO         = "727823TUAM007"
STUDENT_NAME    = "Student Name"          # <-- replace with your name
EXPERIMENT_NAME = f"SKCT_{ROLL_NO}_EEGSeizure"

# SQLite backend — avoids file store corruption and FutureWarning
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mlflow.db")
mlflow.set_tracking_uri(f"sqlite:///{DB_PATH}")


def load_demo_data():
    np.random.seed(42)
    n, sig_len = 100, 256 * 10
    normal  = np.random.randn(n, sig_len) * 0.5
    seizure = np.random.randn(n, sig_len) * 2.0 + \
              np.sin(np.linspace(0, 20 * np.pi, sig_len)) * 3
    return np.vstack([normal, seizure]), np.array([0] * n + [1] * n)


def build_features(signals, labels):
    X, y = [], []
    for signal, label in zip(signals, labels):
        feats = extract_all(preprocess(signal))
        X.append(feats)
        y.extend([label] * len(feats))
    return np.vstack(X), np.array(y)


def run_experiment(model, model_mod, params, run_name, seed, algorithm):
    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name=run_name) as run:
        # Tags — visible in Tags tab
        mlflow.set_tags({
            "student_name": STUDENT_NAME,
            "roll_number":  ROLL_NO,
            "dataset":      "EEGSeizure",
            "algorithm":    algorithm,
        })

        signals, labels = load_demo_data()
        X, y = build_features(signals, labels)
        X_scaled = StandardScaler().fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=seed, stratify=y
        )

        # Params — visible in Params columns
        mlflow.log_param("algorithm",   algorithm)
        mlflow.log_param("random_seed", seed)
        mlflow.log_params(params)

        metrics, trained_model = model_mod.train_evaluate(model, X_train, X_test, y_train, y_test)

        mlflow.log_metrics({
            "f1_score":              metrics["f1_score"],
            "precision":             metrics["precision"],
            "recall":                metrics["recall"],
            "accuracy":              metrics["accuracy"],
            "training_time_seconds": metrics["training_time_seconds"],
        })

        model_path = f"model_{run_name}.pkl"
        model_size_mb = model_mod.save(trained_model, model_path)
        mlflow.log_metric("model_size_mb", model_size_mb)
        mlflow.log_artifact(model_path)
        mlflow.sklearn.log_model(trained_model, name="sklearn_model")
        os.remove(model_path)

        print(f"[{run_name}] F1={metrics['f1_score']:.4f} | "
              f"time={metrics['training_time_seconds']}s | "
              f"size={model_size_mb}MB | run_id={run.info.run_id}")
        return run.info.run_id, metrics["f1_score"]


RF_CONFIGS = [
    {"n_estimators": 50,  "max_depth": 5},
    {"n_estimators": 100, "max_depth": 10},
    {"n_estimators": 150, "max_depth": 15},
    {"n_estimators": 200, "max_depth": None},
    {"n_estimators": 100, "max_depth": 8},
    {"n_estimators": 300, "max_depth": 12},
]

GB_CONFIGS = [
    {"n_estimators": 50,  "learning_rate": 0.1,  "max_depth": 3},
    {"n_estimators": 100, "learning_rate": 0.05, "max_depth": 4},
    {"n_estimators": 150, "learning_rate": 0.01, "max_depth": 5},
    {"n_estimators": 200, "learning_rate": 0.1,  "max_depth": 3},
    {"n_estimators": 100, "learning_rate": 0.2,  "max_depth": 6},
    {"n_estimators": 250, "learning_rate": 0.05, "max_depth": 4},
]

SEEDS = [42, 7, 13, 99, 21, 55]


if __name__ == "__main__":
    results = []

    for i, (cfg, seed) in enumerate(zip(RF_CONFIGS, SEEDS)):
        model = rf_model.build(**cfg, seed=seed)
        run_id, f1 = run_experiment(model, rf_model, cfg, f"RF_run_{i+1}", seed, "RandomForest")
        results.append((run_id, f1, f"RF_run_{i+1}"))

    for i, (cfg, seed) in enumerate(zip(GB_CONFIGS, SEEDS)):
        model = gb_model.build(**cfg, seed=seed)
        run_id, f1 = run_experiment(model, gb_model, cfg, f"GB_run_{i+1}", seed, "GradientBoosting")
        results.append((run_id, f1, f"GB_run_{i+1}"))

    best = max(results, key=lambda x: x[1])
    print(f"\n{'='*50}")
    print(f"Best Run : {best[2]}")
    print(f"Run ID   : {best[0]}")
    print(f"F1 Score : {best[1]:.4f}")
    print(f"Load     : mlflow.sklearn.load_model('runs:/{best[0]}/sklearn_model')")
    print(f"{'='*50}")
