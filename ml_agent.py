import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import List, Optional           


class MLAgent:
    def __init__(
        self,
        feature_list: Optional[List[str]] = None,
        model=None,
    ):
        # Default core set; caller can override
        self.feature_list = feature_list or [
            "close", "volume", "rsi2", "macd", "macd_signal", "sma"
        ]
        self.model = model or LogisticRegression(max_iter=1000)
        self.scaler = StandardScaler()
        self.is_trained = False
        self.validation_accuracy: float | None = None   # for reference

    def train(self, historical_df: pd.DataFrame):
        """
        Train the ML model to predict next-day price movement (up/down).

        Parameters:
        - historical_df: pd.DataFrame with ['close', 'volume', 'rsi2', 'macd', 'macd_signal', 'sma']
        """
        historical_df = historical_df.dropna()

        # Create target: next day up (1) or down (0)
        historical_df['target'] = (historical_df['close'].shift(-1) > historical_df['close']).astype(int)

        features = ['close', 'volume', 'rsi2', 'macd', 'macd_signal', 'sma']
        #X = historical_df[features]
        X = historical_df[self.feature_list]
        y = historical_df['target']

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

        # Normalize features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)

        # Train model
        self.model.fit(X_train_scaled, y_train)
        self.is_trained = True

        # Validation accuracy (optional for printing)
        accuracy = self.model.score(X_val_scaled, y_val)
        print(f"Validation Accuracy: {accuracy:.2f}")
        return accuracy 

    def predict(self, today_row: pd.Series) -> float:
        """
        Predict next-day movement for today's feature row.

        Parameters:
        - today_row: pd.Series with ['close', 'volume', 'rsi', 'macd', 'macd_signal', 'sma']

        Returns:
        - 1 (predicts up) or 0 (predicts down)
        """
        if not self.is_trained:
            raise ValueError("Model not trained yet!")

        #features = ['close', 'volume', 'rsi2', 'macd', 'macd_signal', 'sma']
        #X_today = today_row[features].values.reshape(1, -1)
        X_today = today_row[self.feature_list].values.reshape(1, -1)
        X_today_scaled = self.scaler.transform(X_today)
        pred = self.model.predict_proba(X_today_scaled)[0][1]
        return pred

# Example usage:
if __name__ == "__main__":
    # Simulate sample historical data
    dates = pd.date_range(start="2024-01-01", end="2024-02-10")
    np.random.seed(42)
    prices = np.cumsum(np.random.normal(0, 1, size=len(dates))) + 100

    df = pd.DataFrame({
        "date": dates,
        "close": prices,
        "high": prices + np.random.uniform(0.5, 1.5, size=len(dates)),
        "low": prices - np.random.uniform(0.5, 1.5, size=len(dates)),
        "volume": np.random.randint(1000, 5000, size=len(dates))
    })

    # Add dummy indicators for testing
    df['rsi'] = np.random.uniform(30, 70, size=len(df))
    df['macd'] = np.random.normal(0, 1, size=len(df))
    df['macd_signal'] = np.random.normal(0, 1, size=len(df))
    df['sma'] = df['close'].rolling(window=5).mean()

    agent = MLAgent()
    agent.train(df)

    today_row = df.iloc[-1]
    prediction = agent.predict(today_row)
    print("Predicted next move:", "Up" if prediction == 1 else "Down")
