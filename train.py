"""Train an SMS spam classifier and save the model plus a reference profile for drift monitoring."""
import json, joblib, pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.dummy import DummyClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

df = pd.read_csv("sms.tsv", sep="\t", header=None, names=["label", "text"]).drop_duplicates()
y = (df.label == "spam").astype(int); X = df.text
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
print(f"{len(df)} unique messages, {y.mean():.1%} spam")

def vec(): return TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), min_df=2, lowercase=True)
models = {
 "Baseline (always ham)": make_pipeline(DummyClassifier(strategy="most_frequent")),
 "Naive Bayes": Pipeline([("v", vec()), ("m", MultinomialNB(alpha=0.1))]),
 "Logistic Regression": Pipeline([("v", vec()), ("m", LogisticRegression(max_iter=1000, C=10, class_weight="balanced"))]),
 "Linear SVM (calibrated)": Pipeline([("v", vec()), ("m", CalibratedClassifierCV(LinearSVC(class_weight="balanced"), cv=5))]),
}
rows, fitted = [], {}
for n, m in models.items():
    if n.startswith("Baseline"): m = Pipeline([("v", vec()), ("m", DummyClassifier(strategy="most_frequent"))])
    m.fit(Xtr, ytr); fitted[n] = m
    p = m.predict(Xte); pr = m.predict_proba(Xte)[:, 1]
    rows.append({"model": n, "precision": precision_score(yte, p, zero_division=0), "recall": recall_score(yte, p),
                 "f1": f1_score(yte, p), "roc_auc": roc_auc_score(yte, pr)})
res = pd.DataFrame(rows).round(4); res.to_csv("results.csv", index=False); print(res.to_string(index=False))

# choose the production model by 5-fold CV F1 on the TRAIN split only (test set stays untouched for reporting)
from sklearn.model_selection import cross_val_score
cv = {n: cross_val_score(models[n], Xtr, ytr, cv=5, scoring="f1").mean() for n in models if not n.startswith("Baseline")}
print("\n5-fold CV F1 on train:", {k: round(v, 4) for k, v in cv.items()})
best = max(cv, key=cv.get); print("selected:", best)
m = fitted[best]; p = m.predict(Xte); tn, fp, fn, tp = confusion_matrix(yte, p).ravel()
print(f"\n{best}: false positives (ham flagged as spam) = {fp} of {tn+fp}; missed spam = {fn} of {fn+tp}")
joblib.dump(m, "spam_service/model.joblib")
# reference profile of the training inputs and scores, used by the drift monitor
tr_scores = m.predict_proba(Xtr)[:, 1]
ref = {"length_words": Xtr.str.split().str.len().tolist(), "spam_prob": tr_scores.tolist(),
       "spam_rate": float(ytr.mean())}
json.dump(ref, open("spam_service/reference.json", "w"))
json.dump({"model": best, "test_size": len(yte), "fp": int(fp), "fn": int(fn)}, open("metrics.json", "w"))
