# 727823TUAM007
import argparse
import os
import json
import joblib
import pandas as pd
from datetime import datetime
from sklearn.metrics import (f1_score, accuracy_score, precision_score,
                             recall_score, confusion_matrix, classification_report)

print(f"Roll No: 727823TUAM007 | Timestamp: {datetime.now().isoformat()}")


def main(data_dir, model_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    df    = pd.read_csv(os.path.join(data_dir, "processed_data.csv"))
    X     = df.drop("label", axis=1).values
    y     = df["label"].values
    model = joblib.load(os.path.join(model_dir, "model.pkl"))

    y_pred = model.predict(X)
    report = {
        "f1_score":         round(f1_score(y, y_pred), 4),
        "accuracy":         round(accuracy_score(y, y_pred), 4),
        "precision":        round(precision_score(y, y_pred), 4),
        "recall":           round(recall_score(y, y_pred), 4),
        "confusion_matrix": confusion_matrix(y, y_pred).tolist(),
    }

    report_path = os.path.join(output_dir, "evaluation_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(classification_report(y, y_pred, target_names=["Normal", "Seizure"]))
    print(f"Evaluation complete | F1={report['f1_score']} | Report saved to {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",   default="data_output")
    parser.add_argument("--model_dir",  default="model_output")
    parser.add_argument("--output_dir", default="eval_output")
    args = parser.parse_args()
    main(args.data_dir, args.model_dir, args.output_dir)
