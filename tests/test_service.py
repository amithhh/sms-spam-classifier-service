import os, tempfile
os.environ["SPAM_DB"] = os.path.join(tempfile.mkdtemp(), "t.db")
import numpy as np
from fastapi.testclient import TestClient
from app import app
from spam_service import monitor

c = TestClient(app)

def test_spam_and_ham():
    s = c.post("/predict", json={"text": "WINNER!! You have won a free prize, call 09061701461 now to claim your cash"}).json()
    h = c.post("/predict", json={"text": "Are we still meeting for lunch tomorrow at 1?"}).json()
    assert s["label"] == "spam" and h["label"] == "ham"

def test_validation():
    assert c.post("/predict", json={"text": ""}).status_code == 422
    assert c.post("/predict", json={}).status_code == 422

def test_psi_detects_shift():
    rng = np.random.default_rng(0)
    a = rng.normal(20, 5, 3000)
    assert monitor.psi(a, rng.normal(20, 5, 3000)) < 0.1
    assert monitor.psi(a, rng.normal(35, 5, 3000)) > 0.25

def test_monitor_flags_long_message_traffic():
    for _ in range(60):
        c.post("/predict", json={"text": "please call me back when you get a chance " * 8})
    r = c.get("/monitor").json()
    assert r["status"] == "ok" and r["length_drift"] == "major"
