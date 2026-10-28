"""Part 2, Task D: train a model on the ETL output and save it.

Command line:
    python -m quakes.train --task regression --model linear
    python -m quakes.train --task regression --model poly --degree 3 --alpha 10
    python -m quakes.train --task classification --model logistic [--balanced]

Steps (see the lab handout):
 1. load the table from the SQLite database written by quakes.etl
 2. X = everything except 'mag' and 'big_quake'; y = 'mag' or 'big_quake'
 3. train/test split (stratified for classification, seeded with config.SEED)
 4. fit the dummy baseline AND your model; report both on the test set
 5. cross-validate on the TRAIN set only
 6. classification only: pick the threshold from out-of-fold training
    predictions (cross_val_predict, method='predict_proba'), then apply it to test
 7. save the pipeline to models/ and the report (JSON) to reports/
"""

import argparse
import json
import sqlite3

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_predict, train_test_split

from . import config
from .etl import DB_PATH
from .evaluate import (best_threshold, classification_report_dict,
                       cross_validate_model, regression_report)
from .models import make_classifier, make_regressor

MODELS_DIR = config.ROOT / "models"
REPORTS_DIR = config.ROOT / "reports"


def load_data(db_path=None):
    con = sqlite3.connect(str(db_path or DB_PATH))
    try:
        df = pd.read_sql("SELECT * FROM quakes", con)
    finally:
        con.close()
    # SQLite NULLs come back as None; sklearn imputers want real NaN / numbers
    num = [c for c in config.NUMERIC if c in df.columns]
    df[num] = df[num].apply(pd.to_numeric, errors="coerce")
    for c in config.NOMINAL:
        if c in df.columns:
            df[c] = df[c].astype(object).where(df[c].notna(), np.nan)
    return df


def main() -> None:
    p = argparse.ArgumentParser(description="Train and save a model.")
    p.add_argument("--task", choices=["regression", "classification"], required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--degree", type=int, default=2)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--C", type=float, default=1.0)
    p.add_argument("--balanced", action="store_true")
    args = p.parse_args()

    df = load_data()
    reg = args.task == "regression"
    y = df["mag"] if reg else df["big_quake"]
    X = df.drop(columns=["mag", "big_quake"])          # no target, no leakage

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=config.SEED,
        stratify=None if reg else y,
    )

    report = {"task": args.task, "model": args.model,
              "n_train": len(X_tr), "n_test": len(X_te)}

    if reg:
        base = make_regressor("dummy").fit(X_tr, y_tr)
        model = make_regressor(args.model, args.degree, args.alpha)
        report["params"] = {"degree": args.degree, "alpha": args.alpha}

        cv = cross_validate_model(model, X_tr, y_tr, "neg_root_mean_squared_error")
        model.fit(X_tr, y_tr)

        report["baseline_test"] = regression_report(y_te, base.predict(X_te))
        report["model_test"] = regression_report(y_te, model.predict(X_te))
        report["cv_rmse"] = {"mean": -cv["mean"], "std": cv["std"],
                             "scores": [-s for s in cv["scores"]]}
    else:
        cw = "balanced" if args.balanced else None
        base = make_classifier("dummy").fit(X_tr, y_tr)
        model = make_classifier(args.model, args.C, cw)
        report["params"] = {"C": args.C, "class_weight": cw}

        report["cv_roc_auc"] = cross_validate_model(model, X_tr, y_tr, "roc_auc")

        # threshold from out-of-fold TRAIN predictions only
        oof = cross_val_predict(model, X_tr, y_tr, cv=5, method="predict_proba")[:, 1]
        thr = best_threshold(y_tr, oof, "f1")
        report["threshold"] = thr

        model.fit(X_tr, y_tr)
        p_base = base.predict_proba(X_te)[:, 1]
        p_mod = model.predict_proba(X_te)[:, 1]
        report["baseline_test"] = classification_report_dict(
            y_te, (p_base >= 0.5).astype(int), p_base)
        report["model_test_at_0.5"] = classification_report_dict(
            y_te, (p_mod >= 0.5).astype(int), p_mod)
        report["model_test_at_tuned"] = classification_report_dict(
            y_te, (p_mod >= thr).astype(int), p_mod)

    MODELS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)
    name = f"{args.task}_{args.model}"
    joblib.dump(model, MODELS_DIR / f"{name}.joblib")
    (REPORTS_DIR / f"{name}.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

