import os
from openai import OpenAI
from analysis_agent import AnalysisAgent
from ml_agent import MLAgent
from data_fetch_agent import DataFetchAgent
from fear_greed_agent import get_fear_greed_score
from visualization_agent import VisualizationAgent

import pandas as pd
import numpy as np
import re
import argparse

class LLMCoordinatorAgent:
    def __init__(self, analysis_agent, ml_agent):
        # Read API key from environment variable
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set!")
        
        self.analysis_agent = analysis_agent
        self.ml_agent = ml_agent
        self.client = OpenAI(api_key=api_key)
    def extract_ticker_with_llm(self, user_input: str) -> str:
        system_prompt = (
            "You are a finance assistant. Extract the stock ticker symbol (e.g., AAPL, TSLA, PLTR) "
            "from the user's request. Only reply with the uppercase ticker symbol. "
            "If no ticker is found, reply with 'NONE'."
        )

        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input}
            ],
            temperature=0,
            max_tokens=10,
        )

        extracted = response.choices[0].message.content.strip().upper()
        if re.fullmatch(r"[A-Z]{1,5}", extracted):
            return extracted
        else:
            return self.extract_ticker_regex(user_input)

    def extract_ticker_regex(self, user_input: str) -> str:
        # Fallback if GPT fails
        candidates = re.findall(r'\b[A-Z]{1,5}\b', user_input.upper())
        return candidates[0] if candidates else None

    def generate_summary_with_llm(self, analysis_results: dict, ml_prediction: int, user_input: str,fg_score: float,ml_accuracy: float) -> str:
        system_prompt = (
            "You are a financial analyst assistant. Based on technical indicators and model prediction, "
            "you are given this instruction below to identify overbought: if 2day-RSI is greater than 90, oversold if 2day-RSI is less than 10"
            "summarize the market situation and must suggest a final action."
            "**Important:** At the end of your summary, clearly include this line:\n"
            "Final Probability (UP): <value between 0 and 1>\n"
            "Use that probability to decide BUY (if > 0.5) or SELL (if <= 0.5)."
        )

        user_prompt = (
            f"User asked: {user_input}\n\n"
            f"Here is today's technical analysis:\n"
            f"- 2day-RSI: {analysis_results['rsi2']:.2f}\n"
            f"- 14day-RSI: {analysis_results['rsi14']:.2f}\n"
            f"- MACD Signal: {analysis_results['macd_signal']:.2f}\n"
            #f"- 200d-SMA: {analysis_results['sma']:.2f}\n"
            f"- Support: {analysis_results['support']:.2f}\n"
            f"- Resistance: {analysis_results['resistance']:.2f}\n"
            f"- stoch_k: {analysis_results['stoch_k']:.2f}\n"
            f"- atr: {analysis_results['atr']:.2f}\n"
            f"- upper_band: {analysis_results['bb_uper']:.2f}\n"
            f"- lower_band: {analysis_results['bb_lower']:.2f}\n"
            f"- MA Crossover Signal: {analysis_results.get('ma_crossover')}\n"
            f"- Price vs MA200: {analysis_results.get('price_vs_ma200')}\n"
            f"- Fear & Greed Index: {fg_score} (0 = extreme fear, 100 = extreme greed)\n"
            f"- ML Model Validation Accuracy: {ml_accuracy:.2f}\n"
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

    def handle_request(self, user_input: str, analysis_results: dict, enriched_df: pd.DataFrame,ml_accuracy:float) -> dict:
        today_row = enriched_df.iloc[-1]
        ml_prediction = self.ml_agent.predict(today_row)

        fg_value = today_row.get("fear_greed", None)
        if fg_value is not None:
            analysis_results["fear_greed"] = round(fg_value, 2)

        summary = self.generate_summary_with_llm(analysis_results, ml_prediction, user_input,fg_value,ml_accuracy)

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
    parser = argparse.ArgumentParser(description="Run stock analysis via LLM agent")
    parser.add_argument("user_input", type=str, help="Prompt for the agent (e.g., 'Tell me about AAPL')")
    parser.add_argument("--start", type=str, default=None, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default=None, help="End date (YYYY-MM-DD)")
    args = parser.parse_args()

    user_input = args.user_input
    start_date = args.start
    end_date = args.end

    # Step 1: Create agents
    analysis_agent = AnalysisAgent()
    ml_agent = MLAgent()
    coordinator = LLMCoordinatorAgent(analysis_agent, ml_agent)

    # Step 2: Extract ticker from prompt
    ticker = coordinator.extract_ticker_with_llm(user_input)
    if ticker == "NONE":
        print("Could not extract a valid ticker.")
        exit(1)

    # Step 3: Fetch data
    data_fetch_agent = DataFetchAgent(start=start_date, end=end_date)
    df = data_fetch_agent.fetch(ticker)

    # Step 4: Analyze and enrich
    analysis_results, enriched_df = analysis_agent.analyze(df)

    # Step 5: Merge Fear & Greed Index into enriched_df
    fg_df = get_fear_greed_score(
        start_date or enriched_df["date"].min().strftime("%Y-%m-%d"),
        end_date or enriched_df["date"].max().strftime("%Y-%m-%d")
    )
    print(fg_df.tail)
    enriched_df = pd.merge(enriched_df, fg_df, on="date", how="left")
    enriched_df.rename(columns={"FG": "fear_greed"}, inplace=True)

    print("Latest enriched_df date:", enriched_df["date"].max())

    # Step 6: Train ML model
    ml_accuracy = ml_agent.train(enriched_df)

    # Step 7: Run coordinator
    output = coordinator.handle_request(user_input, analysis_results, enriched_df,ml_accuracy)
    print("Generated Summary:\n", output["summary"])
    viz_agent = VisualizationAgent()
    viz_agent.plot(enriched_df, ticker)