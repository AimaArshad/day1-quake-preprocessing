"""Task 5: run the whole pipeline end to end.

Steps (see LAB_GUIDE.md):
 1. fetch all_week (fall back to config.FALLBACK_PATH if the network fails)
 2. merge any data/stream/*.jsonl rows, then dedupe_latest
 3. clean, engineer features, drop leaky columns
 4. split, fit_transform on train only, transform test
 5. print the report, save train.csv, test.csv, preprocessor.joblib
"""


import argparse
import json
 
import joblib
import pandas as pd
 
from . import config
from .cleaning import clean, dedupe_latest
from .features import (
    add_location_features,
    add_quality_features,
    add_target,
    add_time_features,
    drop_leaky_columns,
    group_rare,
)
from .fetch import fetch_feed, geojson_to_df
from .transform import build_preprocessor, split_data
 
 
def load_raw(feed: str) -> pd.DataFrame:
    """Live feed first; on any failure use the saved fallback file."""
    try:
        df = geojson_to_df(fetch_feed(feed))
        print(f"Fetched {len(df)} rows from the live feed '{feed}'")
    except Exception as e:
        print(f"Live fetch failed ({e}); using fallback file")
        with open(config.FALLBACK_PATH, encoding="utf-8") as f:
            df = geojson_to_df(json.load(f))
        print(f"Loaded {len(df)} rows from {config.FALLBACK_PATH.name}")
    return df
 
 
def load_stream() -> pd.DataFrame:
    """Read every non-empty data/stream/*.jsonl file (empty frame if none)."""
    frames = []
    for p in sorted(config.STREAM_DIR.glob("*.jsonl")):
        if p.stat().st_size > 0:
            frames.append(pd.read_json(p, lines=True, convert_dates=False))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
 
 
def to_dense(a):
    return a.toarray() if hasattr(a, "toarray") else a
 
 
 
def main() -> None:
    parser = argparse.ArgumentParser(description="Run the preprocessing pipeline.")
    parser.add_argument("--feed", default="all_week")
    parser.add_argument("--scaler", default="robust",
                        choices=["standard", "minmax", "robust"])
    args = parser.parse_args()
 
    # 1-2. load, merge stream, keep latest version of each event
    df = load_raw(args.feed)
    stream = load_stream()
    if not stream.empty:
        print(f"Merging {len(stream)} streamed rows")
        df = pd.concat([df, stream], ignore_index=True)
    df = dedupe_latest(df)
    n_loaded = len(df)
 
    # 3. clean, engineer, drop leaky columns
    df = clean(df, verbose=True)
    df = add_time_features(df)
    df = add_quality_features(df)
    df = add_location_features(df)
    df = add_target(df)
    df["region"] = group_rare(df["region"])
    df = drop_leaky_columns(df)
 
    # make sure numeric columns really are numeric (all-None columns arrive as object)
    num = [c for c in config.NUMERIC if c in df.columns]
    df[num] = df[num].apply(pd.to_numeric, errors="coerce")
 
    # 4. split FIRST, then fit on train only
    X_train, X_test, y_train, y_test = split_data(df)
    pre = build_preprocessor(args.scaler)
    Xtr = to_dense(pre.fit_transform(X_train))   # learns medians/scales from train
    Xte = to_dense(pre.transform(X_test))        # only applies what train learned
    names = pre.get_feature_names_out()
 
    # 5. report
    print("\n=== REPORT ===")
    print(f"Rows after load/dedupe : {n_loaded}")
    print(f"Rows after cleaning    : {len(df)}")
    print(f"Train / test rows      : {len(X_train)} / {len(X_test)}")
    print(f"Features after encoding: {Xtr.shape[1]}")
    print(f"big_quake rate (train) : {y_train.mean():.3f}")
    print(f"big_quake rate (test)  : {y_test.mean():.3f}")
    print(f"Scaler                 : {args.scaler}")
    print(f"Leftover NaNs (tr/te)  : {pd.isna(Xtr).sum()} / {pd.isna(Xte).sum()}")
 
    # save
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train = pd.DataFrame(Xtr, columns=names, index=X_train.index)
    train[config.TARGET] = y_train
    test = pd.DataFrame(Xte, columns=names, index=X_test.index)
    test[config.TARGET] = y_test
    train.to_csv(config.PROCESSED_DIR / "train.csv", index=False)
    test.to_csv(config.PROCESSED_DIR / "test.csv", index=False)
    joblib.dump(pre, config.PROCESSED_DIR / "preprocessor.joblib")
    print(f"\nSaved train.csv, test.csv, preprocessor.joblib to {config.PROCESSED_DIR}")
 
 


if __name__ == "__main__":
    main()
