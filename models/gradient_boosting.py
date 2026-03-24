from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import time
import os
import joblib


def build(n_estimators=100, learning_rate=0.1, max_depth=3, seed=42):
    return GradientBoostingClassifier(
        n_estimators=n_estimators, learning_rate=learning_rate,
        max_depth=max_depth, random_state=seed
    )


def train_evaluate(model, X_train, X_test, y_train, y_test):
    t0 = time.time()
    model.fit(X_train, y_train)
    training_time = round(time.time() - t0, 4)

    y_pred = model.predict(X_test)
    return {
        "f1_score":              round(f1_score(y_test, y_pred), 4),
        "accuracy":              round(accuracy_score(y_test, y_pred), 4),
        "precision":             round(precision_score(y_test, y_pred), 4),
        "recall":                round(recall_score(y_test, y_pred), 4),
        "training_time_seconds": training_time,
    }, model


def save(model, path):
    joblib.dump(model, path)
    return round(os.path.getsize(path) / (1024 * 1024), 4)
