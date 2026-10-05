"""Task 1: fetch the USGS GeoJSON feed and turn it into a DataFrame."""
import pandas as pd
import requests
BASE = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary"


def fetch_feed(feed: str = "all_week") -> dict:
    """GET {BASE}/{feed}.geojson and return the parsed dict.

    Use a timeout and raise_for_status(). A misspelled feed name returns
    HTTP 200 with a plain-text body, so raise a clear error if the body
    is not valid JSON.
    """
    url=f"{BASE}/{feed}.geojson"
    r= requests.get(url,timeout=30)
    print("request....\n",r)
    r.raise_for_status()
    try:
        return r.json()
    except ValueError:
        raise ValueError(f"Feed '{feed}' did not return JSON: {r.text[:200]}")
    


def geojson_to_df(payload: dict) -> pd.DataFrame:
    """One row per event.

    Columns: 'id' (top level of each feature), every key in 'properties',
    plus 'lon', 'lat', 'depth_km' from geometry.coordinates = [lon, lat, depth].
    """
    
    rows=[]
    for f in payload['features']:
        # print(f)
        lon, lat, depth=f["geometry"]["coordinates"]
        rows.append({
            "id":f["id"],
            **f["properties"],
                "lon":lon,
                "lat":lat,
                "depth_km":depth
        })
    return pd.DataFrame(rows)

if __name__ == "__main__":
    payload = fetch_feed("all_day")
 
    # Explore: first feature and total count
    print("Number of features:", len(payload["features"]))
    print("First feature:\n", payload["features"][0])
 
    df = geojson_to_df(payload)
    print("\nShape:", df.shape)
    print(df[["id", "mag", "place", "lon", "lat", "depth_km"]].head())
 
    # Missing-value summary: which columns are mostly empty?
    missing = df.isna().mean().sort_values(ascending=False) * 100
    print("\nMissing values (% per column):")
    print(missing.round(1))


# timeout is the maximum time (in seconds) requests waits for the server before raising an error, so your script never hangs forever on a stuck connection.

# raise_for_status() turns HTTP error codes (404, 500, etc.) into a Python exception, so a failed request stops loudly instead of you silently processing a bad response.