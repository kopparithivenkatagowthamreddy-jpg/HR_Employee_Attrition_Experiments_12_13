"""Run monitoring, drift detection and model retraining for HR attrition."""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
for script in ("monitoring_drift.py", "retrain_model.py"):
    print("\nRunning", script)
    subprocess.run([sys.executable, str(HERE / script)], check=True)
print("\nExperiment 12 completed successfully.")
