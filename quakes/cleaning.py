"""Task 3: cleaning."""
import pandas as pd
try:
    from quakes.fetch import fetch_feed, geojson_to_df
except ModuleNotFoundError:
    from fetch import fetch_feed, geojson_to_df

def epoch_ms_to_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """Convert 'time' and 'updated' (epoch milliseconds) to UTC datetimes."""
    df = df.copy()
    for col in ("time", "updated"):
        df[col] = pd.to_datetime(df[col], unit="ms", utc=True)
    return df


def dedupe_latest(df: pd.DataFrame) -> pd.DataFrame:
    """One row per 'id', keeping the row with the greatest 'updated'."""
    return (
        df.sort_values("updated")
        .drop_duplicates(subset="id", keep="last")
        .reset_index(drop=True)
    )


def keep_earthquakes(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise 'type' (strip, lowercase) and keep only 'earthquake'."""
    df = df.copy()
    df["type"] = df["type"].astype("string").str.strip().str.lower()
    return df[df["type"] == "earthquake"].reset_index(drop=True)


def extract_region(place: pd.Series) -> pd.Series:
    """Text after the last comma, or the whole string if there is no comma.
    Missing places become 'Unknown'."""
    
    region = place.astype("string").str.split(",").str[-1].str.strip()
    region = region.replace("", pd.NA)  # e.g. a place ending in a comma
    return region.fillna("Unknown")


def iqr_outlier_mask(s: pd.Series, k: float = 1.5) -> pd.Series:
    """True where a value lies outside [Q1 - k*IQR, Q3 + k*IQR]."""
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - k * iqr, q3 + k * iqr
    return (s < lower) | (s > upper)


def drop_missing_target(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows where 'mag' is missing."""
    return df.dropna(subset=["mag"]).reset_index(drop=True)


def clean(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """Chain the steps above (think about the order) and add a 'region' column."""
    steps = [
        ("epoch_ms_to_datetime", epoch_ms_to_datetime),
        ("dedupe_latest", dedupe_latest),
        ("keep_earthquakes", keep_earthquakes),
        ("drop_missing_target", drop_missing_target),
    ]
    for name, fn in steps:
        before = len(df)
        df = fn(df)
        if verbose:
            print(f"{name:22s} {before:6d} -> {len(df):6d}  (removed {before - len(df)})")
 
    df = df.copy()
    df["region"] = extract_region(df["place"])
    return df




 
if __name__ == "__main__":
    from quakes.fetch import fetch_feed, geojson_to_df
 
    raw = geojson_to_df(fetch_feed("all_month"))
 
    # Explore
    print(raw["type"].value_counts(dropna=False), "\n")
    print(raw["place"].sample(5, random_state=0).tolist(), "\n")
    print(raw[["mag", "depth_km"]].describe(), "\n")
 
    out = clean(raw, verbose=True)
 
    # Decision point: outliers and negative values
    for col in ("mag", "depth_km"):
        mask = iqr_outlier_mask(out[col])
        print(f"\n{col}: {mask.sum()} IQR outliers, {(out[col] < 0).sum()} negative values")