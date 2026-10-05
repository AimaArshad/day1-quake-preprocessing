"""Task 4: feature engineering."""
import pandas as pd

try:  # works with `python -m quakes.features`
    from quakes import config
except ImportError:  # works if config.py sits next to this file
    import config


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add 'hour' and 'dayofweek' (UTC) from 'time'."""
    df = df.copy()
    t = pd.to_datetime(df["time"], utc=True)  # safe even if already datetime
    df["hour"] = t.dt.hour
    df["dayofweek"] = t.dt.dayofweek  # Monday=0 ... Sunday=6
    return df


def add_quality_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add:
    update_lag_hours = hours between 'time' and 'updated'
    is_reviewed      = 1 if status == 'reviewed' else 0
    nst_missing      = 1 if 'nst' is missing else 0
    """
    df = df.copy()
    lag = pd.to_datetime(df["updated"], utc=True) - pd.to_datetime(df["time"], utc=True)
    df["update_lag_hours"] = lag.dt.total_seconds() / 3600
    df["is_reviewed"] = (df["status"] == "reviewed").astype(int)
    df["nst_missing"] = df["nst"].isna().astype(int)
    return df


def add_location_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add 'abs_lat' and 'is_shallow' (1 if depth_km < 70 else 0)."""
    df = df.copy()
    df["abs_lat"] = df["lat"].abs()
    df["is_shallow"] = (df["depth_km"] < 70).astype(int)
    return df


def group_rare(s: pd.Series, top_k: int = 15) -> pd.Series:
    """Keep the top_k most frequent values; replace all others with 'Other'."""
    top = s.value_counts().nlargest(top_k).index
    return s.where(s.isin(top), "Other")
 


def add_target(df: pd.DataFrame) -> pd.DataFrame:
    """Add 'big_quake' = 1 if mag >= 4.5 else 0."""
    df = df.copy()
    df["big_quake"] = (df["mag"] >= 4.5).astype(int)
    return df


def drop_leaky_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop every column listed in config.LEAKY (ignore ones that are absent)."""
    cols = [c for c in config.LEAKY if c in df.columns]
    return df.drop(columns=cols)
