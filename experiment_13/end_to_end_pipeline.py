"""Experiment 13: End-to-end MLOps workflow for HR Employee Attrition."""
from pathlib import Path
import json
import subprocess
import sys
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "outputs/experiment13_end_to_end_report.json"

def main():
    print("=" * 65)
    print("EXPERIMENT 13 - END-TO-END HR MLOPS PIPELINE")
    print("=" * 65)

    steps = []
    monitor = ROOT / "experiment_12/run_experiment12.py"
    result = subprocess.run([sys.executable, str(monitor)], capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        raise RuntimeError("Experiment 12 monitoring/retraining stage failed.")
    steps.append({"stage": "Monitoring, drift detection and retraining", "status": "SUCCESS"})

    model = joblib.load(ROOT / "models/hr_employee_attrition_model.pkl")
    X_test = pd.read_csv(ROOT / "data/processed/X_test.csv")
    y_test = pd.read_csv(ROOT / "data/processed/y_test.csv").squeeze().astype(int)
    predictions = model.predict(X_test)

    evaluation = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1_score": float(f1_score(y_test, predictions, zero_division=0))
    }
    steps.append({"stage": "Model loading and evaluation", "status": "SUCCESS"})

    report = {
        "experiment": 13,
        "project": "HR Employee Attrition Prediction",
        "target_column": "left",
        "workflow": [
            "Load HR employee dataset",
            "Use preprocessed training and testing data",
            "Monitor model performance and feature drift",
            "Retrain and compare candidate model",
            "Promote candidate only when F1-score does not decrease",
            "Evaluate final production model"
        ],
        "stages": steps,
        "final_model_metrics": evaluation,
        "overall_status": "SUCCESS"
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=4))
    print("Final model metrics:", evaluation)
    print("End-to-end report:", REPORT)
    print("EXPERIMENT 13 COMPLETED SUCCESSFULLY")

if __name__ == "__main__":
    main()
