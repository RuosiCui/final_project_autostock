from analysis_agent import AnalysisAgent
from ml_agent import MLAgent
from data_fetch_agent import DataFetchAgent
from fear_greed_agent import get_fear_greed_score
from feature_resolver import extract_features
from typing import Optional
import pandas as pd
import numpy as np
import os
import re
from datetime import datetime, timedelta
import json

def generate_market_summary(client, analysis_results, ml_accuracy,ml_prediction,ticker, hold_days,fg_score):
    """Generate a market summary using OpenAI API directly"""
    system_prompt = (
        "You are a financial analyst assistant. Analyze the market situation based on all available data including technical indicators, "
        "market sentiment, and machine learning predictions. When the ML model accuracy is low (below 0.55), give it less weight in your analysis.\n\n"
        "Use these guidelines for your analysis:\n"
        "- Consider a stock overbought if 2day-RSI is greater than 90\n"
        "- Consider a stock oversold if 2day-RSI is less than 10\n"
        "- Evaluate moving average crossovers and price vs long-term moving averages\n"
        "- Consider stochastic, Bollinger Bands, and other volatility indicators\n"
        "- Factor in the Fear & Greed Index for market sentiment not don't give too much weight to final decision making\n\n"
        "Summarize the market situation and provide your final recommendation.\n\n"
        "**Important:** At the very end of your summary, you must include ONLY these two lines and NOTHING ELSE after them:\n"
        "**Final Probability (UP):** <your calculated probability between 0 and 1 based on ALL factors, not just the ML model>\n"
        "**Recommendation:** BUY or SELL (use BUY if probability > 0.5, SELL if probability <= 0.5)\n\n"
        "Ensure your Final Probability and Recommendation are consistent with each other and with your overall analysis. "
        "DO NOT add any additional text, explanations, or summaries after the Recommendation line."
    )

    
    user_prompt = (
        f"User asked: Market summary for {ticker}\n\n"
        f"Objective: Predict whether this stock will go up or down over the next {hold_days} day(s).\n\n"
        f"TECHNICAL INDICATORS:\n"
        f"- 2day-RSI: {analysis_results['rsi2']:.2f} (>90 = overbought, <10 = oversold)\n"
        f"- 14day-RSI: {analysis_results['rsi14']:.2f} (>70 = overbought, <30 = oversold)\n"
        f"- MACD Signal: {analysis_results['macd_signal']:.2f}\n"
        f"- MACD Histogram: {analysis_results.get('macd_hist', 0):.2f}\n"
        f"- Stochastic K: {analysis_results['stoch_k']:.2f} (>80 = overbought, <20 = oversold)\n"
        f"- ATR (volatility): {analysis_results['atr']:.2f}\n"
        f"- Bollinger Upper Band: {analysis_results['bb_uper']:.2f}\n"
        f"- Bollinger Lower Band: {analysis_results['bb_lower']:.2f}\n"
        f"- Support: {analysis_results['support']:.2f}\n"
        f"- Resistance: {analysis_results['resistance']:.2f}\n"
        f"\nTREND SIGNALS:\n"
        f"- MA Crossover Signal: {analysis_results.get('ma_crossover', 'No data')}\n"
        f"- Price vs MA200: {analysis_results.get('price_vs_ma200', 'No data')}\n"
        f"\nMARKET SENTIMENT:\n"
        f"- Fear & Greed Index: {fg_score} (0-25 = extreme fear, 25-50 = fear, 50-75 = greed, 75-100 = extreme greed)\n"
        f"\nML MODEL PREDICTION:\n"
        f"- Validation Accuracy: {ml_accuracy:.2f} (note: accuracy below 0.55 indicates limited reliability)\n"
        f"- Predicted Probability of UP: {ml_prediction:.2f}\n"
        f"\nProvide a concise market analysis based on ALL the above factors. Remember that your final probability should reflect your overall assessment, not just the ML model prediction. End your analysis with ONLY the Final Probability and Recommendation lines."
    )

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.5,
        max_tokens=500,
    )

    print("\n=== Market Summary ===\n")
    print(response.choices[0].message.content)

    return response.choices[0].message.content

def get_default_dates():
    """Get default date range (past 5 years to today)"""
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=5*365)).strftime("%Y-%m-%d")
    return start_date, end_date

def interactive_run_backtest(hold_days: int = 1, api_key: Optional[str] = None):
    """Interactive version of run_backtest with user inputs and ML model verification"""
    print("=== Interactive Backtest Tool ===")
    print("This tool will run a backtest of the trading strategy on historical data.")
    
    # # Get OpenAI API key
    # api_key = input("Please enter your OpenAI API key: ").strip()
    if not api_key:
        print("Error: API key is required.")
        return
    
    # Get ticker
    ticker = input("Enter ticker symbol (default: NVDA): ").strip().upper()
    if not ticker:
        ticker = "NVDA"
    
    # Get date range
    default_start, default_end = get_default_dates()
    start_date = input(f"Enter start date (YYYY-MM-DD) (default: {default_start}): ").strip()
    if not start_date:
        start_date = default_start
    
    end_date = input(f"Enter end date (YYYY-MM-DD) (default: {default_end}): ").strip()
    if not end_date:
        end_date = default_end
    
    # Get lookback window size
    default_lookback = 3700
    lookback_input = input(f"Enter lookback window size in days (default: {default_lookback}): ").strip()
    try:
        lookback_window = int(lookback_input) if lookback_input else default_lookback
        if lookback_window <= 0:
            print(f"Invalid lookback window size. Using default: {default_lookback}")
            lookback_window = default_lookback
    except ValueError:
        print(f"Invalid input. Using default lookback window size: {default_lookback}")
        lookback_window = default_lookback
    
    # Get indicators for ML training
    default_indicators = "close, volume, rsi2, macd, macd_signal, sma200"
    indicators_input = input(f"Enter indicators for ML training (comma-separated) (default: {default_indicators}): ").strip()
    
    if not indicators_input:
        indicators_input = default_indicators
    
    # Extract canonical indicator names
    raw_indicators = [ind.strip() for ind in indicators_input.split(",")]
    indicators = []
    for indicator in raw_indicators:
        extracted = extract_features(indicator)
        if extracted:
            indicators.extend(extracted)
        else:
            indicators.append(indicator)
    
    print(f"Using indicators: {indicators}")
    
    # Initialize agents
    analysis_agent = AnalysisAgent()
    ml_agent = MLAgent(feature_list=indicators,hold_days=hold_days)
    data_fetch_agent = DataFetchAgent(start=start_date, end=end_date)
    
    # Initialize OpenAI client for market summary generation
    from openai import OpenAI
    openai_client = OpenAI(api_key=api_key)
    
    # Fetch data
    print(f"\nFetching data for {ticker} from {start_date} to {end_date}...")
    df_full = data_fetch_agent.fetch(ticker)
    print(f"Fetched {len(df_full)} days of data.")
    
    # Get Fear & Greed data
    print("Fetching Fear & Greed index data...")
    fg_df = get_fear_greed_score(start_date, end_date)
    df_full = pd.merge(df_full, fg_df, on="date", how="left")
    df_full.rename(columns={"FG": "fear_greed"}, inplace=True)
    
    # Analyze full dataset
    print("\nAnalyzing data and calculating technical indicators...")
    analysis_results, enriched_df = analysis_agent.analyze(df_full)
    
    # Verify indicators exist in the dataset
    missing_indicators = [ind for ind in indicators if ind not in enriched_df.columns]
    if missing_indicators:
        print(f"\nWARNING: The following indicators were not found in the dataset: {missing_indicators}")
        valid_indicators = [ind for ind in indicators if ind in enriched_df.columns]
        if not valid_indicators:
            print("Error: No valid indicators available. Please try again with different indicators.")
            return
        print(f"Proceeding with valid indicators: {valid_indicators}")
        ml_agent = MLAgent(feature_list=valid_indicators,hold_days=hold_days)
    
    # We'll verify the ML model on the initial training period instead of the full dataset
    # This is moved to after the initial training window setup
    
    # Set up backtest parameters
    # lookback_window is now set by user input
    if len(df_full) <= lookback_window:
        print(f"Error: Not enough data for backtest. Need at least {lookback_window} days, but only have {len(df_full)}.")
        return
    
    # Train initial ML model on first lookback_window days
    print("\n=== Initial ML Model Training ===")
    print(f"Training ML model on first {lookback_window} days of data...")
    
    # Make sure we have at least lookback_window days of data
    initial_end_idx = min(lookback_window, len(df_full) - 1)
    initial_df = df_full.iloc[:initial_end_idx].copy()
    initial_analysis_results, initial_enriched_df = analysis_agent.analyze(initial_df)
    
    if len(initial_enriched_df) < 300:
        print(f"Error: Not enough data for initial ML training after preprocessing. Need at least 300 days, but only have {len(initial_enriched_df)}.")
        return
    
    ml_accuracy = ml_agent.train(initial_enriched_df)
    print(f"Initial ML Model Accuracy: {ml_accuracy:.4f}")
    
    # Loop to allow the user to try different indicators if accuracy is low
    while ml_accuracy < 0.55:
        print("\nWARNING: Initial ML model accuracy is very low (barely better than random).")
        proceed = input("Do you want to proceed with the backtest anyway? (y/n): ").strip().lower()
        
        if proceed == 'y':
            break
        
        retry = input("Would you like to try different indicators? (y/n): ").strip().lower()
        if retry != 'y':
            print("Backtest cancelled.")
            return
        
        # Get new indicators
        default_indicators = "close, volume, rsi2, macd, macd_signal, sma200"
        indicators_input = input(f"Enter new indicators for ML training (comma-separated) (default: {default_indicators}): ").strip()
        
        if not indicators_input:
            indicators_input = default_indicators
        
        # Extract canonical indicator names
        raw_indicators = [ind.strip() for ind in indicators_input.split(",")]
        indicators = []
        for indicator in raw_indicators:
            extracted = extract_features(indicator)
            if extracted:
                indicators.extend(extracted)
            else:
                indicators.append(indicator)
        
        print(f"Using indicators: {indicators}")
        
        # Verify indicators exist in the dataset
        missing_indicators = [ind for ind in indicators if ind not in enriched_df.columns]
        if missing_indicators:
            print(f"\nWARNING: The following indicators were not found in the dataset: {missing_indicators}")
            valid_indicators = [ind for ind in indicators if ind in enriched_df.columns]
            if not valid_indicators:
                print("Error: No valid indicators available. Please try again with different indicators.")
                continue
            print(f"Proceeding with valid indicators: {valid_indicators}")
            indicators = valid_indicators
        
        # Create a new ML agent with the new indicators
        ml_agent = MLAgent(feature_list=indicators, hold_days=hold_days)
        
        # Retrain the model
        print("\n=== Retraining ML Model ===")
        print(f"Training ML model on first {lookback_window} days of data with new indicators...")
        ml_accuracy = ml_agent.train(initial_enriched_df, validate=True)
        print(f"New ML Model Accuracy: {ml_accuracy:.4f}")
    
    print(f"\n=== Starting Backtest ===")
    print(f"Using initial training window of {lookback_window} days")
    print(f"Will test on {len(df_full) - lookback_window - hold_days} trading days")
    
    # Ask for confirmation
    proceed = input("Ready to start backtest? This may take a while. (y/n): ").strip().lower()
    if proceed != 'y':
        print("Backtest cancelled.")
        return
    
    # Run backtest
    results = []
    for i in range(lookback_window, len(df_full) - hold_days):  # Leave one row for ground truth
        current_date = df_full.iloc[i]["date"].strftime("%Y-%m-%d")
        print(f"\nProcessing day {i-lookback_window+1} of {len(df_full)-lookback_window-hold_days}: {current_date}")
        
        # Use a rolling window that includes the initial lookback_window days plus all days up to current day
        df_partial = df_full.iloc[:i+1].copy()
        analysis_results, enriched_df = analysis_agent.analyze(df_partial)
        if enriched_df.shape[0] < 300:
            print("Skipping: Not enough data for training")
            continue  # Skip if too few rows to train
        
        # Train ML model on all available data up to current day
        ml_agent.train(enriched_df, validate=False)
        today_row = enriched_df.iloc[-1]
        ml_prediction = ml_agent.predict(today_row)
        print(f"ML prediction: {ml_prediction:.4f}")
        
        # Generate market summary using OpenAI directly
        summary = generate_market_summary(
            openai_client,
            analysis_results,
            ml_accuracy,
            ml_prediction,
            ticker,
            hold_days,
            fg_df.loc[fg_df['date'] <= current_date].iloc[-1]['FG'] if not fg_df.empty else 0
        )
        
        # Extract LLM's final probability - looking for the exact format "**Final Probability (UP):** 0.45"
        prob_match = re.search(r"\*\*Final Probability \(UP\)\:\*\*\s*([0-9]*\.?[0-9]+)", summary)
        llm_prob = float(prob_match.group(1)) if prob_match else None
        
        # Extract recommendation - looking for the exact format "**Recommendation:** SELL"
        rec_match = re.search(r"\*\*Recommendation\:\*\*\s*(BUY|SELL)", summary)
        recommendation = rec_match.group(1) if rec_match else None
        
        # Handle case where regex didn't match
        if llm_prob is None or recommendation is None:
            print("Warning: Could not extract probability or recommendation from summary.")
            
            # Try alternative regex patterns with more flexibility
            if llm_prob is None:
                # Try without asterisks
                alt_prob_match = re.search(r"Final Probability.*?[:：]\s*([0-9]*\.?[0-9]+)", summary)
                if alt_prob_match:
                    llm_prob = float(alt_prob_match.group(1))
                    print(f"Found probability using alternative pattern: {llm_prob}")
            
            if recommendation is None:
                # Try without asterisks
                alt_rec_match = re.search(r"Recommendation:\s*(BUY|SELL)", summary)
                if alt_rec_match:
                    recommendation = alt_rec_match.group(1)
                    print(f"Found recommendation using alternative pattern: {recommendation}")
            
            # If still not found, look for the last occurrence of BUY or SELL in the text
            if recommendation is None:
                buy_matches = [m.start() for m in re.finditer(r"BUY", summary)]
                sell_matches = [m.start() for m in re.finditer(r"SELL", summary)]
                
                if buy_matches and (not sell_matches or buy_matches[-1] > sell_matches[-1]):
                    recommendation = "BUY"
                    print(f"Found recommendation by last occurrence: {recommendation}")
                elif sell_matches:
                    recommendation = "SELL"
                    print(f"Found recommendation by last occurrence: {recommendation}")
            
            # If still not found, use defaults
            if llm_prob is None:
                llm_prob = 0.5
                print(f"Using default probability: {llm_prob}")
            
            if recommendation is None:
                recommendation = "BUY" if llm_prob >= 0.5 else "SELL"
                print(f"Using default recommendation based on probability: {recommendation}")
        
        llm_decision = 1 if recommendation == "BUY" else 0
        
        # Actual price movement (next day vs today)
        actual = int(df_full.iloc[i+hold_days]["close"] > df_full.iloc[i]["close"])
        correct = llm_decision == actual
        
        result = {
            "date": df_full.iloc[i]["date"],
            "llm_prob": llm_prob,
            "recommendation": recommendation,
            "llm_decision": llm_decision,
            "actual": actual,
            "correct": correct
        }
        
        results.append(result)
        
        # Show running accuracy
        running_acc = np.mean([r["correct"] for r in results])
        print(f"Decision: {recommendation} (Prob: {llm_prob:.2f})")
        print(f"Actual movement: {'UP' if actual == 1 else 'DOWN'}")
        print(f"Correct: {correct}")
        print(f"Running accuracy: {running_acc:.4f} ({sum([r['correct'] for r in results])}/{len(results)})")
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Calculate final metrics
    if len(results_df) > 0:
        # Basic accuracy
        final_acc = np.mean(results_df["correct"])
        
        # Calculate confusion matrix metrics
        true_positives = sum((results_df["llm_decision"] == 1) & (results_df["actual"] == 1))
        true_negatives = sum((results_df["llm_decision"] == 0) & (results_df["actual"] == 0))
        false_positives = sum((results_df["llm_decision"] == 1) & (results_df["actual"] == 0))
        false_negatives = sum((results_df["llm_decision"] == 0) & (results_df["actual"] == 1))
        
        # Calculate precision, recall, and F1 score
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Calculate specificity and balanced accuracy
        specificity = true_negatives / (true_negatives + false_positives) if (true_negatives + false_positives) > 0 else 0
        balanced_acc = (recall + specificity) / 2
        
        print(f"\n=== Backtest Complete ===")
        print(f"Total predictions: {len(results_df)}")
        print(f"Correct predictions: {sum(results_df['correct'])}")
        print(f"Final accuracy: {final_acc:.4f}")
        
        print("\n=== Confusion Matrix ===")
        print(f"True Positives (correctly predicted UP): {true_positives}")
        print(f"True Negatives (correctly predicted DOWN): {true_negatives}")
        print(f"False Positives (predicted UP but was DOWN): {false_positives}")
        print(f"False Negatives (predicted DOWN but was UP): {false_negatives}")
        
        print("\n=== Classification Metrics ===")
        print(f"Precision (accuracy of positive predictions): {precision:.4f}")
        print(f"Recall (sensitivity, true positive rate): {recall:.4f}")
        print(f"Specificity (true negative rate): {specificity:.4f}")
        print(f"F1 Score (harmonic mean of precision and recall): {f1_score:.4f}")
        print(f"Balanced Accuracy (average of recall and specificity): {balanced_acc:.4f}")
        
        # Add metrics to results DataFrame for CSV export
        metrics_df = pd.DataFrame({
            'Metric': ['Accuracy', 'Precision', 'Recall', 'Specificity', 'F1 Score', 'Balanced Accuracy',
                      'True Positives', 'True Negatives', 'False Positives', 'False Negatives', 'Total Predictions'],
            'Value': [final_acc, precision, recall, specificity, f1_score, balanced_acc,
                     true_positives, true_negatives, false_positives, false_negatives, len(results_df)]
        })
        
        # Save results to CSV
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"backtest_{ticker}_{timestamp}.csv"
        
        # Save both results and metrics
        results_df.to_csv(filename, index=False)
        metrics_filename = f"metrics_{ticker}_{timestamp}.csv"
        metrics_df.to_csv(metrics_filename, index=False)
        
        print(f"\nResults saved to {filename}")
        print(f"Metrics saved to {metrics_filename}")
    else:
        print("No valid backtest results were generated.")
    
    return results_df

if __name__ == "__main__":
    interactive_run_backtest()