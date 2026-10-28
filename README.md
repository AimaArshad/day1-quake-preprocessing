# Earthquake Data Pipeline & ML Modeling

An end-to-end data engineering and machine learning project built on the live [USGS earthquake feed](https://earthquake.usgs.gov/earthquakes/feed/v1.0/geojson.php). It ingests real-time GeoJSON data, cleans it, engineers leakage-free features, loads it into SQLite through a logged and idempotent ETL job, and trains models that **estimate earthquake magnitude (regression)** and **detect big quakes, magnitude ≥ 4.5 (classification)**.

> **Note:** every event in the USGS feed has already happened. The models *estimate* how large an event was from its location, depth and measurement-quality signals. They do not forecast future earthquakes.

---

## Highlights

- **Robust ingestion**: REST API client with timeouts, HTTP error handling, validation of malformed feed names, and an offline fallback file.
- **Streaming simulation**: a polling collector that de-duplicates events by `(id, updated)` and appends to JSON Lines.
- **Data cleaning**: epoch-millisecond conversion, "latest version wins" de-duplication, filtering non-earthquake events, region extraction, IQR outlier analysis.
- **Leakage prevention**: 11 target-leaking columns (`sig`, `title`, `mmi`, `felt`, `alert`, `tsunami`, `magType`, ...) are removed, preprocessing is fitted on training data only, and the classification threshold is tuned on out-of-fold training predictions.
- **Production-style ETL**: `logging` to file and console, and an idempotent SQLite upsert so hourly reruns never duplicate rows.
- **scikit-learn Pipelines**: imputation, scaling, one-hot encoding and optional polynomial features travel with the model.
- **From-scratch model**: gradient-descent linear regression written as an sklearn-compatible estimator.
- **Honest evaluation**: every model is compared with a dummy baseline, with k-fold cross-validation on the training set only.
- **Inference**: live-feed prediction with back-testing against real USGS values, plus an interactive prompt for manual input.
- **Tests**: a pytest suite for the ETL, models and metrics.

---

## Project structure

```
.
├── quakes/
│   ├── config.py       # paths, seed, target, NUMERIC / NOMINAL / LEAKY column lists
│   ├── fetch.py        # USGS API -> dict -> DataFrame
│   ├── stream.py       # polling collector -> data/stream/*.jsonl
│   ├── cleaning.py     # dates, de-duplication, earthquake filter, region, outliers
│   ├── features.py     # time / quality / location features, target, leaky-column removal
│   ├── transform.py    # train/test split and ColumnTransformer preprocessor
│   ├── run.py          # Day 1 pipeline: fetch -> clean -> split -> CSV + preprocessor
│   ├── etl.py          # extract -> transform -> load into SQLite (logged, idempotent)
│   ├── models.py       # GDLinearRegression, make_regressor, make_classifier
│   ├── evaluate.py     # metrics, threshold search, cross-validation
│   ├── train.py        # train, evaluate, save model + JSON report
│   └── predict.py      # live-feed and interactive predictions
├── tests/              # pytest suites (+ conftest.py with a toy_df fixture)
├── data/               # fallback/, stream/, processed/ (git-ignored)
├── models/             # saved pipelines (.joblib, git-ignored)
├── reports/            # JSON metrics and prediction CSVs
├── logs/               # etl.log
├── environment.yml
└── README.md
```

---

## Architecture

```
            ┌────────────┐    ┌───────────────┐
USGS API ──▶│  fetch.py  │◀───│   stream.py   │  (polls every 60 s -> data/stream/*.jsonl)
            └─────┬──────┘    └───────┬───────┘
                  ▼                   ▼
            ┌──────────────────────────────────┐
            │ cleaning.py  ->  features.py     │   config.py supplies column lists and paths
            └─────┬──────────────────┬─────────┘
                  ▼                  ▼
           run.py (CSV files)   etl.py (SQLite: data/processed/quakes.db)
                                     │
                                     ▼
              train.py  ◀── models.py (Pipeline) + evaluate.py (metrics, CV)
                  │
                  ▼
        models/*.joblib + reports/*.json ──▶ predict.py (live / interactive)
```

---

## Setup

```bash
conda env create -f environment.yml
conda activate niai-ml
pytest
```

Always run modules from the project root with `python -m quakes.<name>`. The package uses relative imports.

---

## Usage

### 1. Collect streaming data (optional)

```bash
python -m quakes.stream --interval 60 --duration 2400
```

### 2. Day 1 preprocessing pipeline

```bash
python -m quakes.run --scaler robust
```

Writes `train.csv`, `test.csv` and `preprocessor.joblib` to `data/processed/`.

### 3. ETL into SQLite

```bash
python -m quakes.etl --mode batch    # live feed only
python -m quakes.etl --mode merge    # live feed + stream files
```

Logs row counts at every stage to `logs/etl.log`. Rerunning is safe: rows are upserted on `id`.

### 4. Train models

```bash
python -m quakes.train --task regression --model linear
python -m quakes.train --task regression --model poly --degree 3 --alpha 10
python -m quakes.train --task classification --model logistic
python -m quakes.train --task classification --model logistic --balanced
```

Each run saves a fitted pipeline to `models/` and a metrics report to `reports/`.

### 5. Predict

```bash
python -m quakes.predict                              # live all_hour feed, back-tested against USGS values
python -m quakes.predict --feed all_day --unseen-only # only events not used in training
python -m quakes.predict --interactive                # enter features by hand
```

### Scheduling the ETL hourly

**Linux/macOS (cron):**

```
0 * * * * cd /path/to/project && /path/to/miniconda3/envs/niai-ml/bin/python -m quakes.etl --mode batch >> logs/cron.log 2>&1
```

**Windows (Task Scheduler):**

```powershell
schtasks /create /tn "QuakesETL" /tr "\"C:\path\to\project\run_etl.bat\"" /sc hourly /mo 1 /f
```

where `run_etl.bat` changes into the project folder and runs `python -m quakes.etl --mode batch`. Use the full path to the environment's Python, since schedulers do not load conda.

---

## Results

Dataset: 1,840 events, split 80/20 (1,472 train / 368 test, stratified for classification, seed 42). The test set is used once, for final scoring.

### Regression: estimating magnitude

| Metric | Dummy baseline (mean) | Linear model |
|---|---|---|
| MAE | 0.86 | **0.36** |
| RMSE | 1.19 | **0.48** |
| R² | -0.003 | **0.83** |

5-fold CV RMSE on the training set: 0.465 ± 0.030, consistent with the test RMSE.

### Classification: detecting big quakes (magnitude ≥ 4.5)

Only about 6% of events are positive, so accuracy is misleading: the dummy baseline scores 94% accuracy while finding zero big quakes.

| Metric | Baseline | Logistic @ 0.5 | Logistic @ tuned threshold (0.2) |
|---|---|---|---|
| Accuracy | 0.940 | 0.976 | 0.981 |
| Precision | 0.000 | 0.783 | 0.800 |
| Recall | 0.000 | 0.818 | **0.909** |
| F1 | 0.000 | 0.800 | **0.851** |
| ROC-AUC | 0.500 | 0.996 | 0.996 |

5-fold CV ROC-AUC: 0.994 ± 0.001. The threshold was selected from out-of-fold training predictions, so the test set played no part in choosing it.

> Exact numbers depend on the feed snapshot and the saved model/report. Re-run training to reproduce your own.

---

## Key design decisions

- **Leakage control**: `config.LEAKY` lists every column that encodes magnitude or is only known after the event. Preprocessing lives inside the Pipeline, so imputers and scalers are refitted within each cross-validation fold.
- **Event identity**: USGS revises events, so `id` alone is not unique over time. The collector keys on `(id, updated)`, and batch processing keeps the latest version per `id`.
- **Idempotent load**: delete-then-append upsert on `id`, so repeated runs are safe and newer versions replace older ones.
- **Class imbalance**: precision, recall, F1 and ROC-AUC are reported next to accuracy, with optional `class_weight="balanced"` and threshold tuning.
- **Cardinality**: `region` is reduced to the top 15 categories plus `"Other"` before one-hot encoding.
- **Inference consistency**: live predictions reuse the training category list for `region` and the saved pipeline for all transforms, so serving matches training.

---

## Limitations

- The target is magnitude *estimation* for events that have already occurred, not earthquake forecasting.
- The test set contains only about 22 positive (big-quake) events, so classification metrics have high variance.
- Part of the classifier's strength comes from legitimate proxies (reporting network, station count, region, review status) that separate small local quakes from large global ones.
- Live-feed evaluation can overlap with training rows. Use `--unseen-only` for an honest check.

---

## Tech stack

Python 3.11 · pandas · NumPy · scikit-learn · SQLite · requests · joblib · pytest · argparse · logging

## Possible extensions

- Rolling counts of recent events per region (`events_last_24h_in_region`)
- Gradient boosting and calibrated probabilities
- A small web dashboard or REST endpoint for predictions
- Dockerfile and CI to run the tests on every push

## Author

*Aima Arshad* · [LinkedIn](https://linkedin.com/in/aima-arshad) · [GitHub](https://github.com/AimaArshad)
