import os
from openai import OpenAI
from analysis_agent import AnalysisAgent
from ml_agent import MLAgent
import pandas as pd
import numpy as np

class LLMCoordinatorAgent:
    def __init__(self, analysis_agent, ml_agent):
        # Read API key from environment variable
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set!")
        
        self.analysis_agent = analysis_agent
        self.ml_agent = ml_agent
        self.client = OpenAI(api_key=api_key)

    def generate_summary_with_llm(self, analysis_results: dict, ml_prediction: int, user_input: str) -> str:
        system_prompt = (
            "You are a financial analyst assistant. Based on technical indicators and model prediction, "
            "summarize the market situation and suggest an action."
        )

        user_prompt = (
            f"User asked: {user_input}\n\n"
            f"Here is today's technical analysis:\n"
            f"- RSI: {analysis_results['rsi']:.2f}\n"
            f"- MACD: {analysis_results['macd']:.2f}\n"
            f"- SMA: {analysis_results['sma']:.2f}\n"
            f"- Support: {analysis_results['support']:.2f}\n"
            f"- Resistance: {analysis_results['resistance']:.2f}\n"
            f"\nThe machine learning model predicts: "
            f"{'UP (suggesting BUY)' if ml_prediction == 1 else 'DOWN (suggesting SELL)'}\n"
            f"\nPlease write a natural language summary and suggest a final action."
        )

        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.5,
            max_tokens=300,
        )

        return response.choices[0].message.content

    def handle_request(self, user_input: str, historical_df: pd.DataFrame) -> dict:
        analysis_results = self.analysis_agent.analyze(historical_df)
        today_row = historical_df.iloc[-1]
        ml_prediction = self.ml_agent.predict(today_row)

        summary = self.generate_summary_with_llm(analysis_results, ml_prediction, user_input)

        decision = "BUY" if ml_prediction == 1 else "SELL"

        return {
            "summary": summary,
            "decision": decision,
            "raw_analysis": analysis_results,
            "raw_prediction": ml_prediction
        }

# ------------------------------------------------------------
# Example Usage
# ------------------------------------------------------------
if __name__ == "__main__":
    # Simulate historical data
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

    df['rsi'] = np.random.uniform(30, 70, size=len(df))
    df['macd'] = np.random.normal(0, 1, size=len(df))
    df['macd_signal'] = np.random.normal(0, 1, size=len(df))
    df['sma'] = df['close'].rolling(window=5).mean()

    analysis_agent = AnalysisAgent()
    ml_agent = MLAgent()
    ml_agent.train(df)

    coordinator = LLMCoordinatorAgent(analysis_agent, ml_agent)
    output = coordinator.handle_request("Tell me about TQQQ", df)

    print("Final Decision:", output["decision"])
    print("Generated Summary:\n", output["summary"])
