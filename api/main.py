
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from pathlib import Path
import pandas as pd
import joblib

# Load the trained HR Attrition model
MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "models"
    / "hr_employee_attrition_model.pkl"
)

model = joblib.load(MODEL_PATH)

app = FastAPI(
    title="HR Employee Attrition Prediction API",
    description="API to predict employee attrition using a trained ML model.",
    version="1.0"
)

# Input details
class EmployeeData(BaseModel):
    satisfaction_level: float = Field(ge=0, le=1)
    last_evaluation: float = Field(ge=0, le=1)
    number_project: int = Field(ge=1)
    average_montly_hours: int = Field(ge=0)
    time_spend_company: int = Field(ge=1)
    work_accident: int = Field(ge=0, le=1)
    promotion_last5years: int = Field(ge=0, le=1)
    department: str
    salary: str


# Home page
@app.get("/")
def home():
    return {
        "message": "HR Employee Attrition Prediction API is running!",
        "documentation": "/docs"
    }


# Health check
@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": True
    }


# Prediction endpoint
@app.post("/predict")
def predict(data: EmployeeData):

    departments = [
        "randd", "accounting", "hr", "management",
        "marketing", "product_mng", "sales",
        "support", "technical", "it"
    ]

    department = data.department.strip().lower()
    department = department.replace(" ", "_")

    if department in ["r&d", "research_and_development"]:
        department = "randd"

    if department not in departments:
        raise HTTPException(
            status_code=400,
            detail="Invalid department. Use IT, R&D, accounting, hr, management, marketing, product_mng, sales, support or technical."
        )

    salary = data.salary.strip().lower()

    if salary not in ["low", "medium", "high"]:
        raise HTTPException(
            status_code=400,
            detail="Salary must be low, medium or high."
        )

    # Create all 18 model features in the correct order
    features = {
        "satisfaction_level": data.satisfaction_level,
        "last_evaluation": data.last_evaluation,
        "number_project": data.number_project,
        "average_montly_hours": data.average_montly_hours,
        "time_spend_company": data.time_spend_company,
        "Work_accident": data.work_accident,
        "promotion_last_5years": data.promotion_last5years,
        "Department_RandD": int(department == "randd"),
        "Department_accounting": int(department == "accounting"),
        "Department_hr": int(department == "hr"),
        "Department_management": int(department == "management"),
        "Department_marketing": int(department == "marketing"),
        "Department_product_mng": int(department == "product_mng"),
        "Department_sales": int(department == "sales"),
        "Department_support": int(department == "support"),
        "Department_technical": int(department == "technical"),
        "salary_low": int(salary == "low"),
        "salary_medium": int(salary == "medium")
    }

    input_data = pd.DataFrame([features])

    prediction = int(model.predict(input_data)[0])

    probabilities = model.predict_proba(input_data)[0]
    classes = list(model.classes_)

    attrition_probability = float(
        probabilities[classes.index(1)]
    ) if 1 in classes else 0.0

    result = "Likely to Leave" if prediction == 1 else "Likely to Stay"

    return {
        "prediction": prediction,
        "result": result,
        "attrition_probability": round(attrition_probability * 100, 2),
        "message": "Prediction completed successfully"
    }
