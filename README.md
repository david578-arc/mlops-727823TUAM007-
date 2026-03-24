# mlops-skct-727823TUAM007

**Student Name:** Student Name  <!-- replace -->
**Roll Number:** 727823TUAM007
**Dataset:** EEG Seizure Detection
**Problem Type:** Binary Classification

---

## Project Structure

```
mlops-skct-727823TUAM007/
│
├── models/                             # Shared model definitions (RF + GB)
│   ├── __init__.py
│   ├── random_forest.py                # build(), train_evaluate(), save()
│   └── gradient_boosting.py            # build(), train_evaluate(), save()
│
├── mlflow_component/                   # Component A — MLflow Experiment Tracking
│   ├── data_preprocessing.py           # Bandpass filter, normalize, segment
│   ├── feature_extraction.py           # Band powers + time-domain features
│   ├── mlflow_experiments.py           # 12 runs, 2 algorithms, all tags/metrics
│   └── requirements.txt
│
├── azure_pipeline/                     # Component B — Azure ML Pipeline
│   ├── data_prep.py                    # Stage 1: preprocess → processed_data.csv
│   ├── train_pipeline.py               # Stage 2: train → model.pkl + metrics.csv
│   ├── evaluate.py                     # Stage 3: evaluate → evaluation_report.json
│   ├── pipeline_727823TUAM007.yml      # Azure ML Pipeline YAML
│   └── requirements.txt
│
└── README.md
```

---

## Component A — Run MLflow Experiments & View Output

### Step 1 — Install dependencies
```bash
cd mlflow_component
pip install -r requirements.txt
```

### Step 2 — Run all 12 experiments
```bash
python mlflow_experiments.py
```

Expected terminal output per run:
```
[RF_run_1] F1=0.9823 | time=1.24s | size=0.45MB | run_id=abc123...
[RF_run_2] F1=0.9901 | time=2.11s | size=0.87MB | run_id=def456...
...
==================================================
Best Run : RF_run_4
Run ID   : <run_id>
F1 Score : 0.9934
Load     : mlflow.sklearn.load_model('runs:/<run_id>/sklearn_model')
==================================================
```

### Step 3 — View in MLflow UI
```bash
mlflow ui
```
Then open: **http://127.0.0.1:5000**

| What to check | Where in UI |
|---|---|
| Experiment name `SKCT_727823TUAM007_EEGSeizure` | Left sidebar → Experiments |
| All 12 runs with timestamps | Experiment page → Runs table |
| Tags: student_name, roll_number, dataset | Click any run → Tags tab |
| Metrics: F1, accuracy, precision, recall | Runs table columns |
| Operational: training_time_seconds, model_size_mb | Runs table columns |
| Best model artifact | Click best run → Artifacts tab → sklearn_model |

### Step 4 — Load best model from MLflow
```python
import mlflow.sklearn
model = mlflow.sklearn.load_model("runs:/<best_run_id>/sklearn_model")
predictions = model.predict(X_test)
```

---

## Component B — Run Azure ML Pipeline & View Output

### Option 1 — Local Test (no Azure account needed)

```bash
cd azure_pipeline
pip install -r requirements.txt

# Stage 1 — Data Prep
python data_prep.py --output_dir data_output

# Stage 2 — Train
python train_pipeline.py --input_dir data_output --output_dir model_output

# Stage 3 — Evaluate
python evaluate.py --data_dir data_output --model_dir model_output --output_dir eval_output
```

Expected outputs:
```
data_output/
└── processed_data.csv

model_output/
├── model.pkl
└── metrics.csv

eval_output/
└── evaluation_report.json
```

View evaluation report:
```bash
type eval_output\evaluation_report.json
```
```json
{
  "f1_score": 0.9912,
  "accuracy": 0.9934,
  "precision": 0.9876,
  "recall": 0.9948,
  "confusion_matrix": [[856, 10], [4, 730]]
}
```

---

### Option 2 — Submit to Azure ML

#### Prerequisites
```bash
pip install azure-cli
az extension add --name ml
az login
```

#### Submit pipeline
```bash
cd azure_pipeline
az ml job create \
  --file pipeline_727823TUAM007.yml \
  --workspace-name <your-workspace> \
  --resource-group <your-resource-group>
```

#### View in Azure ML Portal
1. Go to **https://ml.azure.com**
2. Select your workspace
3. Click **Jobs** in the left sidebar
4. Find experiment **SKCT_727823TUAM007_EEGSeizure**
5. Click the pipeline run to see:
   - Stage graph: `data_prep → train_pipeline → evaluate`
   - Each stage: logs, outputs, metrics
   - Download `evaluation_report.json` from the evaluate stage outputs

#### Capture Run ID
```bash
az ml job list --workspace-name <workspace> --resource-group <rg> --query "[0].name"
```
Paste the Run ID into your submission form and README.

**Azure Run ID:** *(paste here after submission)*

---

## Component A — MLflow Results Table

| Run | Algorithm | n_estimators | max_depth | lr | Seed | F1 | Accuracy | Precision | Recall | Train Time (s) | Model Size (MB) |
|-----|-----------|-------------|-----------|------|------|----|----------|-----------|--------|----------------|-----------------|
| RF_run_1 | RandomForest     | 50  | 5    | —    | 42 | — | — | — | — | — | — |
| RF_run_2 | RandomForest     | 100 | 10   | —    | 7  | — | — | — | — | — | — |
| RF_run_3 | RandomForest     | 150 | 15   | —    | 13 | — | — | — | — | — | — |
| RF_run_4 | RandomForest     | 200 | None | —    | 99 | — | — | — | — | — | — |
| RF_run_5 | RandomForest     | 100 | 8    | —    | 21 | — | — | — | — | — | — |
| RF_run_6 | RandomForest     | 300 | 12   | —    | 55 | — | — | — | — | — | — |
| GB_run_1 | GradientBoosting | 50  | 3    | 0.10 | 42 | — | — | — | — | — | — |
| GB_run_2 | GradientBoosting | 100 | 4    | 0.05 | 7  | — | — | — | — | — | — |
| GB_run_3 | GradientBoosting | 150 | 5    | 0.01 | 13 | — | — | — | — | — | — |
| GB_run_4 | GradientBoosting | 200 | 3    | 0.10 | 99 | — | — | — | — | — | — |
| GB_run_5 | GradientBoosting | 100 | 6    | 0.20 | 21 | — | — | — | — | — | — |
| GB_run_6 | GradientBoosting | 250 | 4    | 0.05 | 55 | — | — | — | — | — | — |

> Fill metric values after running `python mlflow_experiments.py`

### Best Model Rationale
Best run selected by highest **F1 score** — appropriate for imbalanced binary classification (seizure detection) where both false negatives (missed seizures) and false positives (false alarms) carry clinical cost.

---

## Component C — Challenges

During development, the following real error was encountered:

```
UserErrorException: ScriptExecution.StreamAccess.NotFound
Message: The script 'data_prep.py' was not found in the snapshot.
InnerException: FileNotFoundError: [Errno 2] No such file or directory: 'data_prep.py'
Traceback (most recent call last):
  File "azureml-setup/context_manager_injector.py", line 127, in execute_with_context
    runpy.run_path(target_script, run_name="__main__")
FileNotFoundError: [Errno 2] No such file or directory: 'data_prep.py'
```

**Root cause:** `code: .` in the pipeline YAML resolved relative to the wrong working directory. Fixed by ensuring all scripts are co-located with the YAML inside `azure_pipeline/`.

A second issue was MLflow autolog conflicting with manual `log_metric` calls causing duplicate metric keys. Resolved by not calling `mlflow.sklearn.autolog()` and logging all metrics manually.
