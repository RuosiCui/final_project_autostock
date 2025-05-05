import pandas as pd
import ta

class AnalysisAgent:
    def __init__(self, rsi_length=2, sma_window=200):
        self.rsi_length = rsi_length
        self.sma_window = sma_window

  
    def analyze(self, historical_df: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
        """
        Analyze historical stock data and compute key indicators.

        Parameters:
        - historical_df: pd.DataFrame with columns ['date', 'close', 'high', 'low', 'volume']

        Returns:
        - results (dict): latest values for RSI, MACD, etc.
        - enriched_df (pd.DataFrame): full DataFrame with all indicators added
        """
        df = historical_df.sort_values('date').copy()

                # Compute indicators
                # ─────────────────────────────────────────────────────────────
        # 1) CORE PRICE‑BASED INDICATORS
        # ─────────────────────────────────────────────────────────────
        df["rsi2"]       = ta.momentum.RSIIndicator(df["close"], window=2).rsi()
        df["rsi14"]      = ta.momentum.rsi(df["close"], window=14)           # same as above, 14‑period

        macd             = ta.trend.MACD(df["close"])
        df["macd"]        = macd.macd()
        df["macd_signal"] = macd.macd_signal()
        df["macd_hist"]   = macd.macd_diff()

        # Simple & exponential MAs
        df["sma20"]   = ta.trend.sma_indicator(df["close"], window=20)
        df["sma50"]   = ta.trend.sma_indicator(df["close"], window=50)
        df["sma200"]  = ta.trend.sma_indicator(df["close"], window=200)
        df["ema12"]   = ta.trend.ema_indicator(df["close"], 12)
        df["ema26"]   = ta.trend.ema_indicator(df["close"], 26)
        df["dist_sma200"] = (df["close"] - df["sma200"]) / df["sma200"]

        # ─────────────────────────────────────────────────────────────
        # 2) VOLATILITY / RANGE
        # ─────────────────────────────────────────────────────────────
        df["atr"]         = ta.volatility.average_true_range(df["high"], df["low"], df["close"])
        df["bb_upper"]    = ta.volatility.bollinger_hband(df["close"])
        df["bb_lower"]    = ta.volatility.bollinger_lband(df["close"])
        df["bb_width"]    = (df["bb_upper"] - df["bb_lower"]) / df["close"]   # normalised width

        # ─────────────────────────────────────────────────────────────
        # 3) MOMENTUM / OSCILLATORS
        # ─────────────────────────────────────────────────────────────
        df["stoch_k"]     = ta.momentum.stoch(df["high"], df["low"], df["close"])
        df["roc10"]       = ta.momentum.roc(df["close"], window=10)           # 10‑day rate of change
        df["cci20"]       = ta.trend.cci(df["high"], df["low"], df["close"], window=20)

        # ─────────────────────────────────────────────────────────────
        # 4) VOLUME / FLOW
        # ─────────────────────────────────────────────────────────────
        df["vol_ratio"]   = df["volume"] / df["volume"].rolling(20).mean()
        df["mfi14"]       = ta.volume.money_flow_index(df["high"], df["low"], df["close"], df["volume"], window=14)
        df["obv"]         = ta.volume.on_balance_volume(df["close"], df["volume"])

        # ─────────────────────────────────────────────────────────────
        # 5) SUPPORT / RESISTANCE + CALENDAR
        # ─────────────────────────────────────────────────────────────
        df["support"]     = df["low"].rolling(20).min()
        df["resistance"]  = df["high"].rolling(20).max()
        df["dow"]         = df["date"].dt.dayofweek
        df["month"]       = df["date"].dt.month
        # ─────────────────────────────────────────────────────────────


        # Drop rows with NaN due to rolling/EMA
        df.dropna(inplace=True)

        latest = df.iloc[-1]
        results = {
            'rsi2': latest['rsi2'],
            'rsi14': latest['rsi14'],
            'macd': latest['macd'],
            'macd_signal': latest['macd_signal'],
            "macd_hist" : latest["macd_hist"],
            'sma': latest['sma200'],
            "sma20": latest["sma20"],
            "sma50": latest["sma50"],
            'support': latest['support'],
            'resistance': latest['resistance'],
            "stoch_k": latest["stoch_k"],
            'atr': latest['atr'],
            'vol_ratio': latest['vol_ratio'],
            'bb_uper': latest['bb_upper'],
            'bb_lower': latest['bb_lower'],
            "bb_width": latest["bb_width"],
            "roc10": latest["roc10"],
            "cci20": latest["cci20"],
            "mfi14": latest["mfi14"],
            'today_vol/20d_avg': latest['vol_ratio']
        }

        results.update(self.detect_ma_signals(df))

        return results, df
    
    def detect_ma_signals(self,df: pd.DataFrame) -> dict:
        result = {}

        # Ensure MAs exist
        if "sma20" not in df.columns:
            df["sma20"] = df["close"].rolling(20).mean()
        if "sma50" not in df.columns:
            df["sma50"] = df["close"].rolling(50).mean()
        if "sma200" not in df.columns:
            df["sma200"] = df["close"].rolling(200).mean()

        # Most recent row
        today = df.iloc[-1]
        yesterday = df.iloc[-2]

        # Crossover detection
        if yesterday["sma20"] < yesterday["sma50"] and today["sma20"] > today["sma50"]:
            result["ma_crossover"] = "Bullish: 20-day MA crossed above 50-day"
        elif yesterday["sma20"] > yesterday["sma50"] and today["sma20"] < today["sma50"]:
            result["ma_crossover"] = "Bearish: 20-day MA crossed below 50-day"
        else:
            result["ma_crossover"] = "No recent crossover between 20d and 50d"

        # Price vs long-term MA
        if today["close"] > today["sma200"]:
            result["price_vs_ma200"] = "Price is above 200-day MA (bullish)"
        else:
            result["price_vs_ma200"] = "Price is below 200-day MA (bearish)"

        return result



# Example usage:
if __name__ == "__main__":
    # Simulate some historical data
    dates = pd.date_range(start="2024-01-01", end="2024-02-10")
    import numpy as np
    np.random.seed(42)
    prices = np.cumsum(np.random.normal(0, 1, size=len(dates))) + 100

    df = pd.DataFrame({
        "date": dates,
        "close": prices,
        "high": prices + np.random.uniform(0.5, 1.5, size=len(dates)),
        "low": prices - np.random.uniform(0.5, 1.5, size=len(dates)),
        "volume": np.random.randint(1000, 5000, size=len(dates))
    })

    agent = AnalysisAgent()
    result = agent.analyze(df)
    print(result)
