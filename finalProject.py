import pandas as pd
import ta
import numpy as np

# Simulate some stock data (close prices)
dates = pd.date_range(start="2024-01-01", end="2024-01-20", freq="D")
np.random.seed(42)
prices = np.cumsum(np.random.normal(0, 1, size=len(dates))) + 100  # random walk around 100

# Create a sample DataFrame
df = pd.DataFrame({
    "date": dates,
    "close": prices,
    "high": prices + np.random.uniform(0.5, 1.5, size=len(dates)),
    "low": prices - np.random.uniform(0.5, 1.5, size=len(dates)),
    "volume": np.random.randint(1000, 5000, size=len(dates))
})

# Set today to 2024-01-14
today = pd.Timestamp("2024-01-14")
historical_df = df[df['date'] <= today].copy()

# Correct computation using ta
historical_df['rsi'] = ta.momentum.RSIIndicator(close=historical_df['close']).rsi()

macd = ta.trend.MACD(close=historical_df['close'])
historical_df['macd'] = macd.macd()
historical_df['macd_signal'] = macd.macd_signal()

historical_df['sma_5'] = ta.trend.SMAIndicator(close=historical_df['close'], window=5).sma_indicator()

# Look at the result
print(historical_df.tail())
