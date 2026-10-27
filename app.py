"""FastAPI service for the SMS spam model with prediction logging and a drift report."""
import time, joblib, os
from functools import lru_cache
from fastapi import FastAPI
from pydantic import BaseModel, Field
from spam_service import monitor

app = FastAPI(title="SMS spam classifier")
THRESHOLD = float(os.environ.get("SPAM_THRESHOLD", 0.5))

class Msg(BaseModel):
    text: str = Field(min_length=1, max_length=2000)

@lru_cache(maxsize=1)
def model(): return joblib.load(os.path.join(os.path.dirname(__file__), "spam_service", "model.joblib"))

@app.get("/health")
def health(): return {"status": "ok", "threshold": THRESHOLD}

@app.post("/predict")
def predict(m: Msg):
    t0 = time.perf_counter()
    prob = float(model().predict_proba([m.text])[0, 1]); label = int(prob >= THRESHOLD)
    ms = (time.perf_counter() - t0) * 1000
    monitor.log(len(m.text.split()), prob, label, ms)
    return {"label": "spam" if label else "ham", "spam_probability": round(prob, 4)}

@app.get("/monitor")
def monitor_report(): return monitor.report()
