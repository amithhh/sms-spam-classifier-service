"""Prediction logging (SQLite) and drift checks using the Population Stability Index (PSI)."""
import json, os, sqlite3, time
import numpy as np

DB = os.environ.get("SPAM_DB", "predictions.db")
REF = os.path.join(os.path.dirname(__file__), "reference.json")

def _con():
    c = sqlite3.connect(DB)
    c.execute("CREATE TABLE IF NOT EXISTS preds (ts REAL, words INTEGER, spam_prob REAL, label INTEGER, latency_ms REAL)")
    return c

def log(words, prob, label, latency_ms):
    with _con() as c:
        c.execute("INSERT INTO preds VALUES (?,?,?,?,?)", (time.time(), words, prob, label, latency_ms))

def psi(reference, current, bins=10):
    """PSI < 0.1 stable, 0.1-0.25 moderate shift, > 0.25 major shift (common rule of thumb)."""
    reference, current = np.asarray(reference, float), np.asarray(current, float)
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(reference, edges)[0] / len(reference)
    c = np.histogram(current, edges)[0] / len(current)
    r, c = np.clip(r, 1e-4, None), np.clip(c, 1e-4, None)
    return float(np.sum((c - r) * np.log(c / r)))

def report(min_rows=50):
    ref = json.load(open(REF))
    with _con() as c:
        rows = c.execute("SELECT words, spam_prob, label, latency_ms FROM preds").fetchall()
    if len(rows) < min_rows:
        return {"status": "not_enough_data", "rows": len(rows), "needed": min_rows}
    words, prob, label, lat = map(np.array, zip(*rows))
    p_len, p_prob = psi(ref["length_words"], words), psi(ref["spam_prob"], prob)
    flag = lambda v: "major" if v > 0.25 else "moderate" if v > 0.1 else "stable"
    return {"status": "ok", "rows": len(rows),
            "psi_message_length": round(p_len, 3), "length_drift": flag(p_len),
            "psi_spam_score": round(p_prob, 3), "score_drift": flag(p_prob),
            "spam_rate_live": round(float(label.mean()), 3), "spam_rate_training": round(ref["spam_rate"], 3),
            "p95_latency_ms": round(float(np.percentile(lat, 95)), 2)}
