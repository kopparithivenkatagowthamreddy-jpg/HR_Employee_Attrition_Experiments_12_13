"""Experiment 12: Retrain and compare a candidate HR attrition model."""
from pathlib import Path
import json
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
X_TRAIN = ROOT / "data/processed/X_train.csv"
X_TEST = ROOT / "data/processed/X_test.csv"
Y_TRAIN = ROOT / "data/processed/y_train.csv"
Y_TEST = ROOT / "data/processed/y_test.csv"
BASE_MODEL = ROOT / "models/hr_employee_attrition_model.pkl"
CANDIDATE = ROOT / "models/hr_employee_attrition_candidate.pkl"
REPORT = ROOT / "outputs/experiment12_retraining_report.json"

def metrics(model, X, y):
    p = model.predict(X)
    return {
        "accuracy": float(accuracy_score(y, p)),
        "precision": float(precision_score(y, p, zero_division=0)),
        "recall": float(recall_score(y, p, zero_division=0)),
        "f1_score": float(f1_score(y, p, zero_division=0))
    }

def main():
    Xtr, Xte = pd.read_csv(X_TRAIN), pd.read_csv(X_TEST)
    ytr = pd.read_csv(Y_TRAIN).squeeze().astype(int)
    yte = pd.read_csv(Y_TEST).squeeze().astype(int)

    baseline = joblib.load(BASE_MODEL)
    candidate = RandomForestClassifier(
        n_estimators=200, max_depth=10, min_samples_split=20,
        min_samples_leaf=10, random_state=42, class_weight="balanced"
    )
    candidate.fit(Xtr, ytr)

    before = metrics(baseline, Xte, yte)
    after = metrics(candidate, Xte, yte)
    promoted = after["f1_score"] >= before["f1_score"]

    joblib.dump(candidate, CANDIDATE)
    if promoted:
        joblib.dump(candidate, BASE_MODEL)

    report = {
        "experiment": 12,
        "project": "HR Employee Attrition Prediction",
        "baseline_model_metrics": before,
        "candidate_model_metrics": after,
        "promotion_decision": "PROMOTED" if promoted else "NOT PROMOTED",
        "promotion_rule": "Candidate F1-score must be equal to or higher than baseline F1-score",
        "candidate_model_path": str(CANDIDATE.relative_to(ROOT)),
        "production_model_updated": bool(promoted)
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=4))
    print("EXPERIMENT 12 - RETRAINING")
    print("Baseline:", before)
    print("Candidate:", after)
    print("Decision:", report["promotion_decision"])
    print("Report saved:", REPORT)

if __name__ == "__main__":
    main()
