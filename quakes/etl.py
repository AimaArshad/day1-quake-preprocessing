"""Part 2, Task A: extract, transform, load, with logging."""
import logging

import pandas as pd

from . import config

log = logging.getLogger("quakes.etl")
DB_PATH = config.PROCESSED_DIR / "quakes.db"
LOG_PATH = config.ROOT / "logs" / "etl.log"


def extract(feed: str = "all_week") -> pd.DataFrame:
    """Fetch the feed as a DataFrame. If the network fails, log a warning and
    use config.FALLBACK_PATH instead."""
    ...


def read_stream_files() -> pd.DataFrame:
    """Concatenate every data/stream/*.jsonl file (empty DataFrame if none)."""
    ...


def transform(raw: pd.DataFrame) -> pd.DataFrame:
    """Day 1 steps in one call: clean, time/quality/location features, target,
    group_rare on 'region', drop leaky columns."""
    ...


def load_sqlite(df: pd.DataFrame, db_path, table: str = "quakes") -> None:
    """Upsert on 'id': running twice must not duplicate rows, and a row with an
    existing id must be replaced by the new version.
    Hints: sqlite3, df.to_sql, DELETE before append. Datetime columns need to
    be converted to text first."""
    ...


def run_etl(mode: str = "batch") -> None:
    """extract -> (merge stream files if mode == 'merge') -> transform -> load.
    Log row counts at every stage with the logging module."""
    ...


def main() -> None:
    """argparse: --mode {batch, merge}. Configure logging to LOG_PATH and the
    console, then call run_etl."""
    ...


if __name__ == "__main__":
    main()
