# 727823TUAM007
import argparse
import os
import sys
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from models import random_forest as rf_model

print(f"Roll No: 727823TUAM007 | Timestamp: {datetime.now().isoformat()}")


def main(input_dir, output_dir, n_estimators, max_depth, seed):
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(os.path.join(input_dir, "processed_data.csv"))
    X  = df.drop("label", axis=1).values
    y  = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )

    model = rf_model.build(n_estimators=n_estimators, max_depth=max_depth, seed=seed)
    metrics, trained_model = rf_model.train_evaluate(model, X_train, X_test, y_train, y_test)
    metrics["random_seed"] = seed

    model_path = os.path.join(output_dir, "model.pkl")
    metrics["model_size_mb"] = rf_model.save(trained_model, model_path)

    pd.DataFrame([metrics]).to_csv(os.path.join(output_dir, "metrics.csv"), index=False)
    print(f"Training complete | F1={metrics['f1_score']} | "
          f"Time={metrics['training_time_seconds']}s | Model saved to {model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir",    default="data_output")
    parser.add_argument("--output_dir",   default="model_output")
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth",    type=int, default=10)
    parser.add_argument("--seed",         type=int, default=42)
    args = parser.parse_args()
    main(args.input_dir, args.output_dir, args.n_estimators, args.max_depth, args.seed)
