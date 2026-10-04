# Customer Churn Prediction with FastAPI

A churn model is only useful once something else can call it. This project trains a churn model on the real IBM Telco dataset, wraps it in a FastAPI service with input validation and an API key, runs the server, tests it with 11 automated checks, measures its speed, and writes the Docker files needed to deploy it.

Built in Google Colab. No LLM and no API key is needed to run it.

## What it does

1. **Trains** a scikit-learn pipeline (one-hot encoding + Gradient Boosting) on 7,043 customers using 10 features. Test AUC is 0.84, and the average predicted churn on the test set is 26.3% against an actual 26.5%, so the probabilities are well calibrated.
2. **Saves** the model with joblib, together with `model_info.json` (AUC, allowed category values, training time, library version, and a note that the risk labels were chosen by hand).
3. **Writes `app.py`**, a FastAPI service with these endpoints:
   - `GET /health`
   - `GET /model-info`
   - `POST /predict`: one customer
   - `POST /predict_batch`: up to 1,000 customers
   - `GET /stats`: requests served and average latency
4. **Validates every input** with Pydantic: ranges for numbers (for example tenure 0 to 100), exact allowed values for categories, and a consistency check that rejects TotalCharges which cannot fit the tenure and monthly charge.
5. **Protects the endpoints** with an `x-api-key` header. The key comes from an `API_KEY` environment variable, with a demo default for the notebook only.
6. **Runs the server** in the background and tests it with real HTTP calls.
7. **Writes `requirements.txt`** pinned to the exact library versions used, plus a **Dockerfile** on the same Python version.

## Results from the run

**11 of 11 automated checks passed:**

| Check | Result |
|---|---|
| Health returns ok | pass |
| Risky customer scores higher than safe customer (0.82 vs 0.02) | pass |
| Probabilities stay between 0 and 1 | pass |
| Negative tenure and an unknown contract return 422 with the field names | pass |
| Missing field returns 422 | pass |
| No API key returns 401 | pass |
| Wrong API key returns 401 | pass |
| Batch keeps order and length | pass |
| Batch of 1,001 returns 413 | pass |
| Inconsistent TotalCharges returns 422 | pass |
| Stats counter increases by one per request | pass |

**Speed (single Colab machine, 100 sequential requests):** p50 18.6 ms, p95 48.3 ms, max 68.9 ms. A batch of 500 customers took 145 ms.

## Example call

```
curl -X POST http://localhost:8000/predict \
  -H "x-api-key: demo-key-123" -H "Content-Type: application/json" \
  -d '{"SeniorCitizen":0,"tenure":2,"MonthlyCharges":95.5,"TotalCharges":191.0,"Contract":"Month-to-month","InternetService":"Fiber optic","OnlineSecurity":"No","TechSupport":"No","PaperlessBilling":"Yes","PaymentMethod":"Electronic check"}'
```

Response: `{"churn_probability": 0.8162, "risk": "High", "latency_ms": 21.64}`

## Limitations

- **The Dockerfile was written but not run inside Colab.** Test it once with `docker build` and `docker run` on your own machine.
- **The latency numbers are for one small model on one machine.** This is not a load test.
- **The default API key is a demo value.** Set the `API_KEY` environment variable for anything real. A production service would also use proper authentication and HTTPS.
- **The risk labels (Low under 0.3, Medium 0.3 to 0.6, High 0.6 and above) were chosen by hand**, not tuned to a business cost.
- **There is no monitoring yet** for data drift or model performance after deployment.

## Next steps

Log predictions to a database, add drift monitoring, and deploy on Render, Railway or Google Cloud Run.

## Tech stack

FastAPI, Uvicorn, Pydantic, scikit-learn, joblib, pandas, requests.

## How to run

1. Open the notebook in Google Colab.
2. Run the cells from top to bottom. No API key is needed.
3. Download `app.py`, `churn_model.joblib`, `model_info.json`, `requirements.txt` and `Dockerfile` from the Colab Files panel if you want to deploy it.

---

Built by **Akshat Kesharwani** | [GitHub](https://github.com/akshatkesharwani-info) | [Portfolio](https://akshatkesharwani-info.github.io/)
