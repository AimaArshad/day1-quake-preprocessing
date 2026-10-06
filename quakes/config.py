from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FALLBACK_PATH = ROOT / "data" / "fallback" / "all_week.geojson"
STREAM_DIR = ROOT / "data" / "stream"
PROCESSED_DIR = ROOT / "data" / "processed"

SEED = 42
TARGET = "big_quake"

# # TODO (Task 4 and 5): fill these in after you have explored the data.
# NUMERIC: list[str] = []   # numeric feature columns
# NOMINAL: list[str] = []   # categorical feature columns (one-hot encoded)
# LEAKY: list[str] = []     # columns that encode the magnitude: must be dropped



# Numeric feature columns (binary 0/1 flags are kept numeric)
NUMERIC: list[str] = [
    "depth_km", "lat", "lon", "abs_lat",       # location
    "hour", "dayofweek",                       # time
    "nst", "dmin", "rms", "gap",               # network / measurement quality
    "update_lag_hours",                        # revision lag
    "is_reviewed", "nst_missing", "is_shallow" # engineered flags
]
 
# Categorical feature columns (one-hot encoded)
NOMINAL: list[str] = ["region", "net"]
 
# Columns that encode the magnitude (or are only known after the event): must be dropped
LEAKY: list[str] = [
    "title",     # text like "M 4.6 - ..." contains the magnitude
    "sig",       # significance score, derived from magnitude
    "mmi",       # shaking intensity, estimated after the event
    "cdi",       # "Did You Feel It?" intensity, reports arrive later
    "felt",      # number of those reports
    "alert",     # PAGER alert level, computed from estimated impact
    "tsunami",   # flag set by bulletin staff after the event
    "mag",       # the target is built from it
    "magType",   # method used to compute the magnitude
    "magError",  # uncertainty of the magnitude
    "magNst",    # stations used for the magnitude
]