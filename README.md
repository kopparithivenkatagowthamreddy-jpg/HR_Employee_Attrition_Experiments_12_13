# HR Employee Attrition - Experiments 12 and 13

This package contains **only Experiment 12 and Experiment 13** adapted from the supplied Churn MLOps reference project to the HR Employee Attrition project.

- Target column: `left` (1 = employee left, 0 = employee stayed)
- Dataset: `data/raw/HR_comma_sep.csv`
- Existing model: `models/hr_employee_attrition_model.pkl`

## Folder contents

- `experiment_12/monitoring_drift.py` - checks model performance, feature drift and prediction distribution.
- `experiment_12/retrain_model.py` - trains a candidate Random Forest, compares F1-score and promotes only if it does not decrease.
- `experiment_12/run_experiment12.py` - runs the Experiment 12 scripts.
- `experiment_13/end_to_end_pipeline.py` - runs the monitoring/retraining workflow and evaluates the final model.
- `data/` - HR dataset and the existing processed train/test files.
- `models/` - existing HR model and generated candidate model.
- `outputs/` - generated monitoring and experiment reports.

## Setup

Open a terminal in this folder and create/activate a virtual environment if desired.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run Experiment 12

```bash
python experiment_12/run_experiment12.py
```

## Run Experiment 13

```bash
python experiment_13/end_to_end_pipeline.py
```

Reports are saved in `outputs/`.

**Note:** The incoming batch used for drift demonstration is simulated from the HR test features. This is suitable for demonstrating monitoring in the lab; it is not a live HR data feed. The candidate model is promoted only when its F1-score is at least the baseline F1-score.
