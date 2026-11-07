"""Experiment 12: Monitor HR model performance and detect data/prediction drift."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib
from scipy.stats import ks_2samp, chi2_contingency
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models/hr_employee_attrition_model.pkl"
X_TEST = ROOT / "data/processed/X_test.csv"
Y_TEST = ROOT / "data/processed/y_test.csv"
REPORT = ROOT / "outputs/experiment12_monitoring_report.json"

def main():
    model = joblib.load(MODEL)
    X = pd.read_csv(X_TEST)
    y = pd.read_csv(Y_TEST).squeeze().astype(int)
    pred = model.predict(X)
    prob = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else pred.astype(float)

    metrics = {
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1_score": float(f1_score(y, pred, zero_division=0)),
    }

    # Simulate a new HR batch by shifting selected workforce measurements.
    incoming = X.copy()
    rng = np.random.default_rng(42)
    if "satisfaction_level" in incoming:
        incoming["satisfaction_level"] = np.clip(
            incoming["satisfaction_level"] - rng.normal(0.12, 0.025, len(incoming)), 0, 1
        )
    if "average_montly_hours" in incoming:
        incoming["average_montly_hours"] = np.clip(
            incoming["average_montly_hours"] + rng.normal(25, 8, len(incoming)), 96, 350
        )
    if "time_spend_company" in incoming:
        incoming["time_spend_company"] = np.clip(
            incoming["time_spend_company"] + rng.integers(0, 2, len(incoming)), 2, 10
        )

    drift = {}
    for col in X.columns:
        if pd.api.types.is_numeric_dtype(X[col]):
            stat, p = ks_2samp(X[col].dropna(), incoming[col].dropna())
            drift[col] = {"test": "KS", "statistic": float(stat), "p_value": float(p),
                          "status": "DRIFT" if p < 0.05 else "STABLE"}
        else:
            base = X[col].astype(str).value_counts(normalize=True)
            new = incoming[col].astype(str).value_counts(normalize=True)
            categories = sorted(set(base.index) | set(new.index))
            observed = np.array([[base.get(c, 0) * len(X), new.get(c, 0) * len(incoming)] for c in categories])
            try:
                chi, p, _, _ = chi2_contingency(observed + 1e-8)
            except Exception:
                chi, p = 0.0, 1.0
            drift[col] = {"test": "chi-square", "statistic": float(chi), "p_value": float(p),
                          "status": "DRIFT" if p < 0.05 else "STABLE"}

    new_pred = model.predict(incoming)
    pred_rate_base = float(np.mean(pred))
    pred_rate_new = float(np.mean(new_pred))
    report = {
        "experiment": 12,
        "project": "HR Employee Attrition Prediction",
        "target": "left",
        "baseline_performance": metrics,
        "prediction_distribution": {
            "baseline_employee_attrition_rate": pred_rate_base,
            "incoming_employee_attrition_rate": pred_rate_new,
            "change": pred_rate_new - pred_rate_base
        },
        "feature_drift": drift,
        "drift_detected": any(v["status"] == "DRIFT" for v in drift.values()),
        "note": "Incoming batch is simulated for academic monitoring demonstration."
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=4))
    print("EXPERIMENT 12 - HR MODEL MONITORING")
    print("Baseline metrics:", metrics)
    print("Features with drift:", [k for k,v in drift.items() if v["status"] == "DRIFT"])
    print("Report saved:", REPORT)

if __name__ == "__main__":
    main()
