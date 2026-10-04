
import json, time, os
from collections import deque
import joblib
import pandas as pd
from typing import List, Literal
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

model = joblib.load("churn_model.joblib")
info = json.load(open("model_info.json"))
API_KEY = os.environ.get("API_KEY", "demo-key-123")

app = FastAPI(title="Churn Prediction API", version=info["version"])
request_log = deque(maxlen=1000)


def to_dict(model_obj):
    if hasattr(model_obj, "model_dump"):
        return model_obj.model_dump()
    return model_obj.dict()


def check_consistent(c):
    if c.TotalCharges > (c.tenure + 1) * c.MonthlyCharges * 2:
        raise HTTPException(status_code=422, detail="TotalCharges is not consistent with tenure and MonthlyCharges")


class Customer(BaseModel):
    SeniorCitizen: int = Field(..., ge=0, le=1)
    tenure: int = Field(..., ge=0, le=100)
    MonthlyCharges: float = Field(..., gt=0, lt=1000)
    TotalCharges: float = Field(..., ge=0)
    Contract: Literal['Month-to-month', 'One year', 'Two year']
    InternetService: Literal['DSL', 'Fiber optic', 'No']
    OnlineSecurity: Literal['No', 'No internet service', 'Yes']
    TechSupport: Literal['No', 'No internet service', 'Yes']
    PaperlessBilling: Literal['No', 'Yes']
    PaymentMethod: Literal['Bank transfer (automatic)', 'Credit card (automatic)', 'Electronic check', 'Mailed check']


def risk_label(p):
    if p >= 0.6:
        return "High"
    if p >= 0.3:
        return "Medium"
    return "Low"


def check_key(key):
    if key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing x-api-key header")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/model-info")
def model_info():
    return info


@app.post("/predict")
def predict(customer: Customer, x_api_key: str = Header(default=None)):
    check_key(x_api_key)
    start = time.time()
    check_consistent(customer)
    row = pd.DataFrame([to_dict(customer)])
    prob = float(model.predict_proba(row)[0, 1])
    ms = round((time.time() - start) * 1000, 2)
    request_log.append({"prob": prob, "ms": ms})
    return {"churn_probability": round(prob, 4), "risk": risk_label(prob), "latency_ms": ms}


@app.post("/predict_batch")
def predict_batch(customers: List[Customer], x_api_key: str = Header(default=None)):
    check_key(x_api_key)
    if len(customers) > 1000:
        raise HTTPException(status_code=413, detail="Max 1000 customers per request")
    for c in customers:
        check_consistent(c)
    rows = pd.DataFrame([to_dict(c) for c in customers])
    probs = model.predict_proba(rows)[:, 1]
    return [{"churn_probability": round(float(p), 4), "risk": risk_label(float(p))} for p in probs]


@app.get("/stats")
def stats():
    n = len(request_log)
    avg = sum(r["ms"] for r in request_log) / n if n > 0 else 0
    return {"requests_served": n, "avg_latency_ms": round(avg, 2)}
