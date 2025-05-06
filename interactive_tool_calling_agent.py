import os
import re
import json
import pandas as pd
import datetime
from openai import OpenAI
from typing import List, Optional, Tuple
from analysis_agent import AnalysisAgent
from ml_agent import MLAgent
from data_fetch_agent import DataFetchAgent
from fear_greed_agent import get_fear_greed_score
from visualization_agent import VisualizationAgent
from feature_resolver import extract_features
from interactive_run_backtest import interactive_run_backtest

class InteractiveToolCallingAgent:
    def __init__(self, analysis_agent, ml_agent, api_key=None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not provided and OPENAI_API_KEY environment variable not set!")

        self.analysis_agent = analysis_agent
        self.ml_agent = ml_agent
        self.viz_agent = VisualizationAgent()
        self.client = OpenAI(api_key=self.api_key)

        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "plot_stock_chart",
                    "description": "Generate interactive stock chart with given date.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "ticker": {"type": "string"},
                            "start_date": {"type": "string", "format": "date"},
                            "end_date": {"type": "string", "format": "date"}
                        },
                        "required": ["ticker"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "train_model",
                    "description": "Train ML model on selected indicators and return validation accuracy.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "ticker": {"type": "string"},
                            "start_date": {"type": "string", "format": "date"},
                            "end_date": {"type": "string", "format": "date"},
                            "indicators": {
                                "type": "array",
                                "items": {"type": "string"}
                            },
                            "hold_days": {
                                "type": "integer",
                                "description": "Number of days to hold the stock for the prediction"
                            }
                        },
                        "required": ["ticker", "hold_days", "indicators"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "analyze_stock",
                    "description": "Run technical indicator analysis and return enriched DataFrame.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "ticker": {"type": "string"},
                            "start_date": {"type": "string", "format": "date"},
                            "end_date": {"type": "string", "format": "date"}
                        },
                        "required": ["ticker"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_fear_greed",
                    "description": "Retrieve CNN Fear & Greed index data between two dates.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "start_date": {"type": "string", "format": "date"},
                            "end_date": {"type": "string", "format": "date"}
                        },
                        "required": []
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "summarize_market_view",
                    "description": "Make a buy/sell prediction based on technical indicators, ML model, and CNN Fear & Greed index. Use when user asks 'Should I buy or sell?'",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "ticker": {"type": "string"},
                            "start_date": {"type": "string"},
                            "end_date": {"type": "string"},
                            "hold_days": {
                                "type": "integer",
                                "description": "Number of days to hold the stock for the prediction"
                            }
                        },
                        "required": ["ticker","hold_days"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "run_backtest",
                    "description": "Run an interactive backtest of the trading strategy with customizable parameters. This will help validate the strategy's performance on historical data.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "hold_days": {
                                "type": "integer",
                                "description": "Number of days to hold the stock before evaluating the outcome"
                            }
                        },
                        "required": ["hold_days"]
                    }
                }
            }
        ]

    def run_llm_with_tools(self, user_input: str, conversation_history=None):
        # Track whether any tool was called
        self.tools_called = []
        
        # Extract potential feature names from user input for later use
        self.extracted_features = extract_features(user_input)
        if self.extracted_features:
            print(f"Detected features in input: {self.extracted_features}")
        if conversation_history is None:
            conversation_history = [
                {"role": "system", "content": "You are a financial assistant. Choose appropriate tools for the user's request."}
            ]
        
        # Add the new user message to the conversation history
        conversation_history.append({"role": "user", "content": user_input})
        
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=conversation_history,
            tools=self.tools,
            tool_choice="auto"
        )

        # Add the assistant's response to the conversation history
        conversation_history.append(response.choices[0].message)

        while response.choices[0].finish_reason == "tool_calls":
            tool_messages = []

            for tool_call in response.choices[0].message.tool_calls:
                fn_name = tool_call.function.name
                args = json.loads(tool_call.function.arguments)
                
                # Track which tools are called
                self.tools_called.append({"name": fn_name, "args": args})
                print(f"\n=== TOOL CALLED: {fn_name} ===")
                print(f"Arguments: {json.dumps(args, indent=2)}")
                
                result_str = self.dispatch_function(fn_name, args)
                tool_messages.append({
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": fn_name,
                    "content": result_str or "Done."
                })

            # Add tool responses to the conversation history
            conversation_history.extend(tool_messages)
            
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=conversation_history,
                tools=self.tools,
                tool_choice="auto"
            )
            
            # Add the assistant's new response to the conversation history
            conversation_history.append(response.choices[0].message)

        # Check if any tools were called
        if not self.tools_called:
            print("\n=== NO TOOLS CALLED ===")
            print("The LLM responded directly without using any tools.")
        else:
            print(f"\n=== TOOLS SUMMARY ===")
            print(f"Total tools called: {len(self.tools_called)}")
            for i, tool in enumerate(self.tools_called, 1):
                print(f"{i}. {tool['name']} with args: {list(tool['args'].keys())}")
        
        # Return both the final response and the updated conversation history
        return response.choices[0].message.content, conversation_history

    def get_default_dates(self) -> Tuple[str, str]:
        """Return default start and end dates"""
        today = datetime.datetime.now().strftime("%Y-%m-%d")
        default_start = "2010-01-01"
        return default_start, today
        
    def prompt_for_dates(self, args: dict) -> Tuple[str, str]:
        """Prompt user for start and end dates if not provided"""
        default_start, default_end = self.get_default_dates()
        
        start_date = args.get("start_date")
        if not start_date:
            print(f"\nStart date not provided. Enter start date (YYYY-MM-DD) or press Enter for default ({default_start}):")
            user_input = input().strip()
            start_date = user_input if user_input else default_start
            
        end_date = args.get("end_date")
        if not end_date:
            print(f"\nEnd date not provided. Enter end date (YYYY-MM-DD) or press Enter for default (today: {default_end}):")
            user_input = input().strip()
            end_date = user_input if user_input else default_end
            
        return start_date, end_date

    def dispatch_function(self, fn_name: str, args: dict):
        if fn_name == "plot_stock_chart":
            print(args)
            self.handle_plot(**args)
            return "Chart generated and saved."
        elif fn_name == "train_model":
            self.handle_train(**args)
            return "Training complete."
        elif fn_name == "analyze_stock":
            result, _ = self.handle_analysis(**args)
            return json.dumps(result, indent=2)
        elif fn_name == "get_fear_greed":
            print(args)
            return self.handle_fear_greed(**args).tail(3).to_string(index=False)
        elif fn_name == "summarize_market_view":
            return self.handle_summary(**args)
        elif fn_name == "run_backtest":
            return self.handle_backtest(**args)
        else:
            return f"Unknown function: {fn_name}"
            
    def handle_backtest(self, hold_days: int = 1):
        """Handle the run_backtest function by calling the interactive_run_backtest function"""
        print("\n=== Starting Interactive Backtest ===")
        print("This will launch a separate interactive session for backtesting.")
        print("The current session will be paused until the backtest is complete.")
        print("Follow the prompts to configure and run your backtest.")
        
        # Call the interactive_run_backtest function
        results_df = interactive_run_backtest(hold_days=hold_days,api_key=self.api_key)
        
        if results_df is not None and not results_df.empty:
            return "Backtest completed successfully. Results have been saved to CSV."
        else:
            return "Backtest was cancelled or did not produce any results."

    def handle_plot(self, ticker: str, start_date: str = None, end_date: str = None):
        # Get dates if not provided
        if start_date is None or end_date is None:
            start_date, end_date = self.prompt_for_dates({"start_date": start_date, "end_date": end_date})
            print(f"Using date range: {start_date} to {end_date}")
        fetcher = DataFetchAgent(start=start_date, end=end_date)
        df = fetcher.fetch(ticker)
        _, enriched_df = self.analysis_agent.analyze(df)
        self.viz_agent.plot(enriched_df, ticker, add_benchmarks=True)

    def handle_train(self, ticker: str, indicators: List[str], start_date: str = None, end_date: str = None, hold_days: int = 1):
        # Get dates if not provided
        if start_date is None or end_date is None:
            start_date, end_date = self.prompt_for_dates({"start_date": start_date, "end_date": end_date})
            print(f"Using date range: {start_date} to {end_date}")
        # Convert user-friendly indicator names to canonical names using feature_resolver
        canonical_indicators = []
        for indicator in indicators:
            # Try to extract canonical names from each indicator string
            extracted = extract_features(indicator)
            if extracted:
                canonical_indicators.extend(extracted)
            else:
                # If no match found, keep the original name
                canonical_indicators.append(indicator)
        
        print(f"Using indicators: {canonical_indicators}")
        
        fetcher = DataFetchAgent(start=start_date, end=end_date)
        df = fetcher.fetch(ticker)
        ml_agent = MLAgent(feature_list=canonical_indicators,hold_days=hold_days)
        _, enriched_df = self.analysis_agent.analyze(df)
        
        # Verify all requested indicators exist in the DataFrame
        missing_indicators = [ind for ind in canonical_indicators if ind not in enriched_df.columns]
        if missing_indicators:
            error_msg = f"The following indicators were not found in the data: {missing_indicators}"
            print(error_msg)
            return f"Error: {error_msg}"
            
        acc = ml_agent.train(enriched_df)
        print(f"ML model trained on {ticker} with indicators {canonical_indicators}. Accuracy: {acc:.2f}")

    def handle_analysis(self, ticker: str, start_date: str = None, end_date: str = None):
        # Get dates if not provided
        if start_date is None or end_date is None:
            start_date, end_date = self.prompt_for_dates({"start_date": start_date, "end_date": end_date})
            print(f"Using date range: {start_date} to {end_date}")
        fetcher = DataFetchAgent(start=start_date, end=end_date)
        df = fetcher.fetch(ticker)
        fg_df = self.handle_fear_greed(start_date, end_date)

        analysis_results, enriched_df = self.analysis_agent.analyze(df)
        enriched_df = pd.merge(enriched_df, fg_df, on="date", how="left")
        return analysis_results, enriched_df

    def handle_fear_greed(self, start_date: str = None, end_date: str = None):
        # Get dates if not provided
        if start_date is None or end_date is None:
            start_date, end_date = self.prompt_for_dates({"start_date": start_date, "end_date": end_date})
            print(f"Using date range: {start_date} to {end_date}")
        return get_fear_greed_score(start_date, end_date)

    def handle_summary(self, ticker: str, start_date: str = None, end_date: str = None, hold_days: int = 1):
        # Get dates if not provided
        if start_date is None or end_date is None:
            start_date, end_date = self.prompt_for_dates({"start_date": start_date, "end_date": end_date})
            print(f"Using date range: {start_date} to {end_date}")
        analysis_results, enriched_df = self.handle_analysis(ticker, start_date, end_date)
        
        # Extract features from the user input if available, otherwise use default features
        features_to_use = self.extracted_features if hasattr(self, 'extracted_features') and self.extracted_features else [
            "close", "volume", "rsi2", "macd", "macd_signal", "sma200"
        ]
        
        print(f"Using indicators for market summary: {features_to_use}")
        
        # Create a new ML agent with the selected features
        temp_ml_agent = MLAgent(feature_list=features_to_use,hold_days=hold_days)
        
        # Verify all requested indicators exist in the DataFrame
        missing_indicators = [ind for ind in features_to_use if ind not in enriched_df.columns]
        if missing_indicators:
            print(f"Warning: The following indicators were not found and will be ignored: {missing_indicators}")
            # Filter out missing indicators
            features_to_use = [ind for ind in features_to_use if ind in enriched_df.columns]
            if not features_to_use:
                # If no valid indicators remain, fall back to default indicators that exist in the DataFrame
                default_indicators = ["close", "volume"]
                available_indicators = [col for col in default_indicators if col in enriched_df.columns]
                if available_indicators:
                    features_to_use = available_indicators
                    print(f"Falling back to basic indicators: {features_to_use}")
                else:
                    return "Error: No valid indicators available for ML prediction."
            
            # Recreate ML agent with valid indicators
            temp_ml_agent = MLAgent(feature_list=features_to_use,hold_days=hold_days)
        
        ml_accuracy = temp_ml_agent.train(enriched_df)
        today_row = enriched_df.iloc[-1]
        ml_prediction = temp_ml_agent.predict(today_row)
        fg_value = today_row.get("FG", 0)

        return self.generate_summary_with_llm(
            analysis_results, ml_prediction,
            f"Market summary for {ticker}",
            fg_value, ml_accuracy
        )

    def generate_summary_with_llm(self, analysis_results: dict, ml_prediction: int, user_input: str, fg_score: float, ml_accuracy: float,hold_days: int = 1,) -> str:
        system_prompt = (
            f"You are a financial analyst assistant. Analyze the market situation based on all available data including technical indicators, "
            f"market sentiment, and machine learning predictions. The goal is to predict the stock movement over the next **{hold_days} day(s)**.\n\n"
            "When the ML model accuracy is low (below 0.55), give it less weight in your analysis. But above 0.55 is considered OK.\n\n"
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
            f"User asked: {user_input}\n\n"
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

        response = self.client.chat.completions.create(
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

def run_interactive_session():
    print("=== Interactive Financial Assistant ===")
    print("Type 'exit' or 'quit' to end the session.")
    
    # Get API key from user
    api_key = input("Please enter your OpenAI API key: ").strip()
    
    # Initialize agents
    analysis_agent = AnalysisAgent()
    ml_agent = MLAgent(feature_list=["close", "volume", "rsi2", "macd", "macd_signal", "sma200"])
    
    try:
        coordinator = InteractiveToolCallingAgent(analysis_agent, ml_agent, api_key=api_key)
        print("API key accepted. Financial assistant is ready.")
        
        # Initialize conversation history
        conversation_history = None
        
        while True:
            user_input = input("\nEnter your request: ").strip()
            
            if user_input.lower() in ['exit', 'quit']:
                print("Exiting session. Goodbye!")
                break
            
            try:
                response, conversation_history = coordinator.run_llm_with_tools(user_input, conversation_history)
                print("\nAssistant:", response)
            except Exception as e:
                print(f"Error: {str(e)}")
                
    except ValueError as e:
        print(f"Error: {str(e)}")
        print("Please restart the program with a valid API key.")
    except Exception as e:
        print(f"Unexpected error: {str(e)}")

if __name__ == "__main__":
    run_interactive_session()