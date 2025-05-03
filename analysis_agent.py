import pandas as pd
import ta

class AnalysisAgent:
    def __init__(self, rsi_length=14, sma_window=50):
        self.rsi_length = rsi_length
        self.sma_window = sma_window

    # def analyze(self, historical_df: pd.DataFrame) -> dict:
    #     """
    #     Analyze historical stock data and compute key indicators.

    #     Parameters:
    #     - historical_df: pd.DataFrame with columns ['date', 'close', 'high', 'low', 'volume']

    #     Returns:
    #     - dict with latest RSI, MACD, SMA, support, resistance
    #     """
    #     results = {}

    #     # Ensure the DataFrame is sorted
    #     historical_df = historical_df.sort_values('date')

    #     # Compute RSI
    #     historical_df['rsi'] = ta.momentum.RSIIndicator(close=historical_df['close'], window=self.rsi_length).rsi()

    #     # Compute MACD and MACD Signal
    #     macd_obj = ta.trend.MACD(close=historical_df['close'])
    #     historical_df['macd'] = macd_obj.macd()
    #     historical_df['macd_signal'] = macd_obj.macd_signal()

    #     # Compute SMA
    #     historical_df['sma'] = ta.trend.SMAIndicator(close=historical_df['close'], window=self.sma_window).sma_indicator()

    #     # Compute Support and Resistance (Rolling Low/High)
    #     window_sr = 20  # You can adjust the support/resistance window
    #     historical_df['support'] = historical_df['low'].rolling(window=window_sr).min()
    #     historical_df['resistance'] = historical_df['high'].rolling(window=window_sr).max()

    #     # Fetch the latest available values
    #     latest = historical_df.iloc[-1]
    #     results['rsi'] = latest['rsi']
    #     results['macd'] = latest['macd']
    #     results['macd_signal'] = latest['macd_signal']
    #     results['sma'] = latest['sma']
    #     results['support'] = latest['support']
    #     results['resistance'] = latest['resistance']

    #     return results
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
        df['rsi'] = ta.momentum.RSIIndicator(close=df['close'], window=self.rsi_length).rsi()
        macd_obj = ta.trend.MACD(close=df['close'])
        df['macd'] = macd_obj.macd()
        df['macd_signal'] = macd_obj.macd_signal()
        df['sma'] = ta.trend.SMAIndicator(close=df['close'], window=self.sma_window).sma_indicator()
        df['support'] = df['low'].rolling(window=20).min()
        df['resistance'] = df['high'].rolling(window=20).max()

        # Drop rows with NaN due to rolling/EMA
        df.dropna(inplace=True)

        latest = df.iloc[-1]
        results = {
            'rsi': latest['rsi'],
            'macd': latest['macd'],
            'macd_signal': latest['macd_signal'],
            'sma': latest['sma'],
            'support': latest['support'],
            'resistance': latest['resistance']
        }

        return results, df


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
