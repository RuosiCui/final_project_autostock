from analysis_agent import AnalysisAgent
from ml_agent import MLAgent
from data_fetch_agent import DataFetchAgent
from llmCoordinatorAgent import LLMCoordinatorAgent
from fear_greed_agent import get_fear_greed_score
import pandas as pd
import numpy as np
import os

def run_backtest(ticker: str, start_date: str, end_date: str):
    analysis_agent = AnalysisAgent()
    ml_agent = MLAgent()
    coordinator = LLMCoordinatorAgent(analysis_agent, ml_agent)
    data_fetch_agent = DataFetchAgent(start=start_date, end=end_date)

    df_full = data_fetch_agent.fetch(ticker)
    print("Number of rows in df_full:", len(df_full))

    # Get full F&G range
    fg_df = get_fear_greed_score(start_date, end_date)
    df_full = pd.merge(df_full, fg_df, on="date", how="left")
    df_full.rename(columns={"FG": "fear_greed"}, inplace=True)

    results = []
    lookback_window = 3700

    for i in range(lookback_window, len(df_full) - 1):  # Leave one row for ground truth
        df_partial = df_full.iloc[:i+1].copy()
        analysis_results, enriched_df = analysis_agent.analyze(df_partial)
        if enriched_df.shape[0] < 300:
            continue  # Skip if too few rows to train

        ml_accuracy = ml_agent.train(enriched_df)

        output = coordinator.handle_request(
            user_input=f"What do you think about {ticker} tomorrow?",
            analysis_results=analysis_results,
            enriched_df=enriched_df,
            ml_accuracy=ml_accuracy
        )

        summary = output["summary"]

        # Extract LLM's final probability if available
        import re
        prob_match = re.search(r"Final Probability.*?[:：]\s*([0-9]*\.?[0-9]+)", summary)
        llm_prob = float(prob_match.group(1)) if prob_match else None
        llm_decision = 1 if llm_prob and llm_prob >= 0.5 else 0

        # Actual price movement (next day vs today)
        actual = int(df_full.iloc[i+1]["close"] > df_full.iloc[i]["close"])

        results.append({
            "date": df_full.iloc[i]["date"],
            "llm_prob": llm_prob,
            "llm_decision": llm_decision,
            "actual": actual
        })
        print(results)

    return pd.DataFrame(results)

# Run example
if __name__ == "__main__":
    df_results = run_backtest(ticker="NVDA", start_date="2010-01-01", end_date="2025-01-01")
    print(df_results.tail())
    acc = np.mean(df_results["llm_decision"] == df_results["actual"])
    print(f"\nBacktest Accuracy: {acc:.2f}")
