# src/etl/fetch_f1_data.py

# Jolpi API in order to get more data 

import requests
import pandas as pd
from pathlib import Path
import time

RAW_PATH = Path("data/raw")
FALLBACK_BASE = "https://api.jolpi.ca/ergast/f1"


def safe_get(url, retries=5, delay=2):
    """GET request with retry logic to avoid temporary failures."""
    for attempt in range(retries):
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            return response
        except Exception as e:
            time.sleep(delay)
    raise RuntimeError(f"❌ Failed to fetch after {retries} attempts: {url}")


def fetch_season(year: int):
    """Fetch F1 race results ONLY from fallback API."""
    url = f"{FALLBACK_BASE}/{year}/results.json?limit=1000&offset=0"
    print(f"Fetching {year} from fallback...")
    response = safe_get(url)
    return response.json()


def flatten_results(data):
    """Flatten Ergast JSON into a DataFrame."""
    races = data["MRData"]["RaceTable"]["Races"]

    rows = []
    for race in races:
        for res in race["Results"]:
            rows.append({
                "season": race["season"],
                "round": int(race["round"]),
                "raceName": race["raceName"],
                "circuit": race["Circuit"]["circuitName"],
                "driver": f"{res['Driver']['givenName']} {res['Driver']['familyName']}",
                "constructor": res["Constructor"]["name"],
                "position": int(res["position"]),
                "points": float(res["points"]),
                "grid": int(res["grid"]),
                "laps": int(res["laps"]),
                "status": res["status"]
            })

    df = pd.DataFrame(rows)
    print(f"📄 Rows flattened: {len(df)}")
    return df


def save_results(df, year):
    RAW_PATH.mkdir(parents=True, exist_ok=True)
    path = RAW_PATH / f"race_results_{year}.csv"
    df.to_csv(path, index=False)
    print(f"💾 Saved {path}")


if __name__ == "__main__":
    for year in range(2015, 2025):
        print("=================================")
        print(f"🚗 Processing season {year}")
        print("=================================")

        data = fetch_season(year)
        df = flatten_results(data)

        # Sanity check
        if len(df) < 300:
            print("⚠️ WARNING: Low row count (<300). Retrying after cooldown...")
            time.sleep(15)
            data = fetch_season(year)
            df = flatten_results(data)

        save_results(df, year)

    print("\n✅ All seasons processed.")
