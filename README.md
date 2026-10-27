# SMS Spam Classifier: Model Serving + Drift Monitoring

A spam classifier wrapped in a REST API that validates input, logs every prediction, and reports when live traffic drifts away from the training data.

Data: UCI SMS Spam Collection (5,169 unique messages after removing duplicates, 12.6% spam), included as `sms.tsv`.

## Model
TF-IDF (1-2 grams) with four candidates. The production model is chosen by 5-fold cross-validated F1 on the **training split only**; the test set is used once, for reporting.

| Model (test set, 1,034 messages) | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| Baseline (always "ham") | 0.000 | 0.000 | 0.000 | 0.500 |
| Naive Bayes | 0.967 | 0.893 | 0.929 | 0.990 |
| Logistic Regression | 0.916 | 0.916 | 0.916 | 0.990 |
| **Calibrated Linear SVM (selected)** | 0.938 | 0.916 | 0.927 | 0.991 |

The selected model flags 8 of 903 legitimate messages as spam and misses 11 of 131 spam messages. For a real inbox, false positives cost more than misses, so the threshold is configurable (`SPAM_THRESHOLD`, default 0.5); raising it trades recall for precision. Differences between the three models are small on a test set this size.

## Service
- `POST /predict` with `{"text": "..."}` returns the label and spam probability. Empty or oversized input is rejected (HTTP 422).
- `GET /monitor` compares logged traffic with the training data using the Population Stability Index (PSI) on message length and on the spam score, and reports live spam rate and p95 latency. PSI above 0.25 is flagged "major", 0.1-0.25 "moderate".
- Predictions are stored in SQLite (`predictions.db`).

**Drift check (simulated):** 400 messages sampled from the dataset gave PSI 0.019 on length ("stable"). The same messages repeated five times each gave PSI 5.04 ("major"), correctly flagged. This is a simulation on the same dataset, not real production traffic, and the score distribution PSI stayed low, which shows length drift can occur without the model's output moving.

## Run
```
pip install -r requirements.txt
python train.py                 # trains, prints the table, writes the model and reference profile
uvicorn app:app --reload        # POST /predict, GET /monitor
pytest tests                    # 4 tests: predictions, validation, PSI, drift flagging
```
A `Dockerfile` is included but I have not built it in the environment I wrote this in. It has not been deployed anywhere.
