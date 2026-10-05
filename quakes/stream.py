"""Task 2: simulate streaming by polling a feed on a timer."""
import pandas as pd
import argparse
import os
import time
from datetime import datetime

try:                            # works with `python -m quakes.stream`
    from quakes.fetch import fetch_feed, geojson_to_df
except ImportError:             # works if fetch.py sits next to this file
    from fetch import fetch_feed, geojson_to_df

def filter_unseen(df: pd.DataFrame, seen: set) -> pd.DataFrame:
    """Return only rows whose (id, updated) pair is not in `seen`.

    Add the new pairs to `seen` (modify the set in place).
    """
    
    if df.empty:
        return df
 
    pairs = list(zip(df["id"], df["updated"]))
    keep = [p not in seen for p in pairs]   # True for rows never seen before
    new_df = df[keep]
    seen.update(p for p, k in zip(pairs, keep) if k)  # in-place update
    return new_df
    


def run_stream(feed: str, interval_s: int, duration_s: int, out_path: str) -> None:
    """Poll `feed` every `interval_s` seconds for `duration_s` seconds.

    Each poll: fetch, parse, keep unseen rows, append them to `out_path`
    as JSON lines, and print e.g. "[14:02:11] 3 new / 12 fetched".
    A failed poll must not stop the loop.
    """
    
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
 
    seen: set = set()
    end_time = time.time() + duration_s
 
    while time.time() < end_time:
        stamp = datetime.now().strftime("%H:%M:%S")
        try:
            payload = fetch_feed(feed)
            df = geojson_to_df(payload)
            new_df = filter_unseen(df, seen)
 
            if not new_df.empty:
                # mode "a" = append, so earlier lines are never overwritten
                with open(out_path, "a", encoding="utf-8") as f:
                    f.write(new_df.to_json(orient="records", lines=True))
                    f.write("\n")
 
            print(f"[{stamp}] {len(new_df)} new / {len(df)} fetched", flush=True)
        except Exception as e:  # one bad poll must not stop the loop
            print(f"[{stamp}] poll failed: {e}", flush=True)
 
        time.sleep(interval_s)


def main() -> None:
    """argparse: --feed (default all_hour), --interval (60), --duration (2400),
    --out (data/stream/stream.jsonl), then call run_stream."""
    
    
    parser = argparse.ArgumentParser(description="Poll the USGS feed and save new events.")
    parser.add_argument("--feed", default="all_hour")
    parser.add_argument("--interval", type=int, default=60)
    parser.add_argument("--duration", type=int, default=2400)
    parser.add_argument("--out", default="data/stream/stream.jsonl")
    args = parser.parse_args()
 
    run_stream(args.feed, args.interval, args.duration, args.out)


if __name__ == "__main__":
    main()
