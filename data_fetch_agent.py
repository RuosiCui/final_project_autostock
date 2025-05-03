import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

class DataFetchAgent:
    def __init__(self, start: str = None, end: str = None, lookback_days: int = 60):
        """
        Initialize the agent with optional date range or fallback to lookback_days.

        Parameters:
        - start (str): start date in 'YYYY-MM-DD' format
        - end (str): end date in 'YYYY-MM-DD' format
        - lookback_days (int): used if start and end are not specified
        """
        self.start = start
        self.end = end
        self.lookback_days = lookback_days

    def fetch(self, ticker: str) -> pd.DataFrame:
        """
        Fetch historical OHLCV data for a given stock ticker.

        Returns a DataFrame with ['date', 'open', 'close', 'high', 'low', 'volume']
        """
        print(self.start)
        print(self.end)

        if self.start and self.end:
            end_dt = pd.to_datetime(self.end)
            end_plus_one = (end_dt + timedelta(days=1)).strftime("%Y-%m-%d")
            df = yf.download(ticker, start=self.start, end=end_plus_one, interval="1d", progress=False)
        else:
            df = yf.download(ticker, period=f"{self.lookback_days}d", interval="1d", progress=False)
        print(df.tail)
        if df.empty:
            raise ValueError(f"No data returned for ticker: {ticker}")

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)


        df.reset_index(inplace=True)
        df.rename(columns={
            "Date": "date",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume"
        }, inplace=True)

        return df[['date', 'open', 'close', 'high', 'low', 'volume']]

# ✅ Optional test
if __name__ == "__main__":
    agent = DataFetchAgent(start="1980-01-01", end="2024-03-31")
    df = agent.fetch("AAPL")
    print(df.head())
