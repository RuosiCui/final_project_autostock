import pandas as pd
import requests
from datetime import datetime
from typing import Optional
import cloudscraper


import os

# Get the directory of the current script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Path to the CSV file in the same directory as the script
HISTORICAL_CSV_PATH = os.path.join(SCRIPT_DIR, "fear_greed_history.csv")
REALTIME_API_URL = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata/"

def load_historical_data() -> pd.DataFrame:
    df = pd.read_csv(HISTORICAL_CSV_PATH)
    df['date'] = pd.to_datetime(df['date'], format="%m/%d/%y")
    return df

def fetch_latest_score() -> pd.DataFrame:
    url = REALTIME_API_URL
    scraper = cloudscraper.create_scraper()
    response = scraper.get(url)

    if response.status_code != 200:
        print("Status Code:", response.status_code)
        print("Response Text:", response.text[:300])
        raise Exception("Failed to fetch latest data from CNN API")

    json_data = response.json()
    data = json_data.get("fear_and_greed")

    if isinstance(data, dict):
        date = pd.to_datetime(data['timestamp']).date()
        score = int(round(data['score']))
        df = pd.DataFrame([{'date': date, 'FG': score}])
    else:
        raise Exception("Unexpected format for 'fear_and_greed' data from API")

    return df



def get_fear_greed_score(start_date: str, end_date: Optional[str] = None) -> pd.DataFrame:
    hist_df = load_historical_data()

    if end_date:
        end = pd.to_datetime(end_date)
    else:
        latest_df = fetch_latest_score()
        latest_df['date'] = pd.to_datetime(latest_df['date'])
        hist_df = pd.concat([hist_df, latest_df]).drop_duplicates('date', keep='last')
        end = latest_df['date'].max()

    start = pd.to_datetime(start_date)
    return hist_df[(hist_df['date'] >= start) & (hist_df['date'] <= end)].sort_values('date').reset_index(drop=True)

if __name__ == "__main__":
    print("=== Test: Historical range with end date ===")
    df1 = get_fear_greed_score("2024-12-31", "2025-01-02")
    print(df1.head())

    print("\n=== Test: Range with latest realtime data (no end date) ===")
    df2 = get_fear_greed_score("2024-12-01")
    print(df2.head())

