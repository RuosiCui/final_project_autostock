import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from visualization_agent import VisualizationAgent

def create_sample_data(days=100):
    """Create sample stock data for testing visualization"""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    # Create date range
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Generate random price data with a trend
    np.random.seed(42)  # For reproducibility
    close = np.random.normal(loc=100, scale=1, size=len(dates))
    # Add trend
    close = close + np.linspace(0, 20, len(dates))
    
    # Generate OHLC data
    high = close + np.random.normal(loc=1, scale=0.5, size=len(dates))
    low = close - np.random.normal(loc=1, scale=0.5, size=len(dates))
    open_price = low + np.random.normal(loc=0.5, scale=0.3, size=len(dates)) * (high - low)
    
    # Generate volume data
    volume = np.random.normal(loc=1000000, scale=200000, size=len(dates))
    volume = np.abs(volume)  # Ensure positive values
    
    # Create DataFrame
    df = pd.DataFrame({
        'date': dates,
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    })
    
    # Calculate indicators
    # RSI-2
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain_2 = gain.rolling(window=2).mean()
    avg_loss_2 = loss.rolling(window=2).mean()
    rs_2 = avg_gain_2 / avg_loss_2
    df['rsi2'] = 100 - (100 / (1 + rs_2))
    
    # RSI-14
    avg_gain_14 = gain.rolling(window=14).mean()
    avg_loss_14 = loss.rolling(window=14).mean()
    rs_14 = avg_gain_14 / avg_loss_14
    df['rsi14'] = 100 - (100 / (1 + rs_14))
    
    # MACD
    df['ema12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['ema26'] = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = df['ema12'] - df['ema26']
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    return df

def main():
    print("Creating sample data...")
    df = create_sample_data(days=100)
    
    print("Initializing visualization agent...")
    viz_agent = VisualizationAgent()
    
    print("Plotting data...")
    # Plot with benchmark tickers (SPY and QQQ)
    viz_agent.plot(df, ticker="SAMPLE", add_benchmarks=True)
    
    print("Done! The plot should be displayed in your browser.")

if __name__ == "__main__":
    main()