# import os
# import re
# import json
# import pandas as pd
# from openai import OpenAI
# from typing import List, Optional
# from analysis_agent import AnalysisAgent
# from ml_agent import MLAgent
# from data_fetch_agent import DataFetchAgent
# from fear_greed_agent import get_fear_greed_score
# from visualization_agent import VisualizationAgent
# from feature_resolver import extract_features

# class ToolCallingAgent:
#     def __init__(self, analysis_agent, ml_agent):
#         api_key = os.getenv("OPENAI_API_KEY")
#         if not api_key:
#             raise ValueError("OPENAI_API_KEY environment variable not set!")

#         self.analysis_agent = analysis_agent
#         self.ml_agent = ml_agent
#         self.viz_agent = VisualizationAgent()
#         self.client = OpenAI(api_key=api_key)

#         self.tools = [
#             {
#                 "type": "function",
#                 "function": {
#                     "name": "plot_stock_chart",
#                     "description": "Generate interactive stock chart with RSI, SMA, and MACD.",
#                     "parameters": {
#                         "type": "object",
#                         "properties": {
#                             "ticker": {"type": "string"},
#                             "start_date": {"type": "string", "format": "date"},
#                             "end_date": {"type": "string", "format": "date"}
#                         },
#                         "required": ["ticker", "start_date", "end_date"]
#                     }
#                 }
#             },
#             {
#                 "type": "function",
#                 "function": {
#                     "name": "train_model",
#                     "description": "Train ML model on selected indicators and return validation accuracy.",
#                     "parameters": {
#                         "type": "object",
#                         "properties": {
#                             "ticker": {"type": "string"},
#                             "start_date": {"type": "string", "format": "date"},
#                             "end_date": {"type": "string", "format": "date"},
#                             "indicators": {
#                                 "type": "array",
#                                 "items": {"type": "string"}
#                             }
#                         },
#                         "required": ["ticker", "start_date", "end_date", "indicators"]
#                     }
#                 }
#             }
#         ]

#     def run_llm_with_tools(self, user_input: str):
#         messages = [
#             {"role": "system", "content": "You are a financial assistant. Choose appropriate tools for the user's request."},
#             {"role": "user", "content": user_input}
#         ]

#         response = self.client.chat.completions.create(
#             model="gpt-4o",
#             messages=messages,
#             tools=self.tools,
#             tool_choice="auto"
#         )

#         while response.choices[0].finish_reason == "tool_calls":
#             tool_messages = []

#             for tool_call in response.choices[0].message.tool_calls:
#                 fn_name = tool_call.function.name
#                 args = json.loads(tool_call.function.arguments)
#                 result_str = self.dispatch_function(fn_name, args)
#                 tool_messages.append({
#                     "tool_call_id": tool_call.id,
#                     "role": "tool",
#                     "name": fn_name,
#                     "content": result_str or "Done."  # must return a string
#                 })

#             messages.append(response.choices[0].message)
#             messages.extend(tool_messages)
#             response = self.client.chat.completions.create(
#                 model="gpt-4o",
#                 messages=messages,
#                 tools=self.tools,
#                 tool_choice="auto"
#             )

#     def dispatch_function(self, fn_name: str, args: dict):
#         if fn_name == "plot_stock_chart":
#             self.handle_plot(**args)
#             return "Chart generated and saved."
#         elif fn_name == "train_model":
#             self.handle_train(**args)
#         else:
#             print(f"Unknown function: {fn_name}")

#     def handle_plot(self, ticker: str, start_date: str, end_date: str):
#         fetcher = DataFetchAgent(start=start_date, end=end_date)
#         df = fetcher.fetch(ticker)
#         _, enriched_df = self.analysis_agent.analyze(df)

#         compare_df_spy = fetcher.fetch("SPY")
#         compare_df_qqq = fetcher.fetch("QQQ")


#         self.viz_agent.plot(enriched_df, ticker)

#     def handle_train(self, ticker: str, start_date: str, end_date: str, indicators: List[str]):
#         if not indicators:
#             print("No indicators provided for training.")
#             return

#         fetcher = DataFetchAgent(start=start_date, end=end_date)
#         df = fetcher.fetch(ticker)
#         ml_agent = MLAgent(feature_list=indicators)
#         _, enriched_df = self.analysis_agent.analyze(df)
#         acc = ml_agent.train(enriched_df)
#         print(f"ML model trained on {ticker} with indicators {indicators}. Accuracy: {acc:.2f}")
#         return
# if __name__ == "__main__":
#     print("=== Tool Calling Agent ===")
#     user_input = input("Enter your request (e.g., 'Show me a chart of AAPL from 2023-01-01 to 2024-01-01'): ").strip()

#     # Default agents
#     analysis_agent = AnalysisAgent()
#     ml_agent = MLAgent(feature_list=["close", "volume", "rsi2", "macd", "macd_signal", "sma200"])

#     coordinator = ToolCallingAgent(analysis_agent, ml_agent)
#     coordinator.run_llm_with_tools(user_input)

# === ToolCallingAgent with Reused handle_analysis in Summary ===
import os
import re
import json
import pandas as pd
from openai import OpenAI
from typing import List, Optional
from analysis_agent import AnalysisAgent
from ml_agent import MLAgent
from data_fetch_agent import DataFetchAgent
from fear_greed_agent import get_fear_greed_score
from visualization_agent import VisualizationAgent
from feature_resolver import extract_features

class ToolCallingAgent:
    def __init__(self, analysis_agent, ml_agent):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set!")

        self.analysis_agent = analysis_agent
        self.ml_agent = ml_agent
        self.viz_agent = VisualizationAgent()
        self.client = OpenAI(api_key=api_key)

        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "plot_stock_chart",
                    "description": "Generate interactive stock chart with RSI, SMA, and MACD.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "ticker": {"type": "string"},
                            "start_date": {"type": "string", "format": "date"},
                            "end_date": {"type": "string", "format": "date"}
                        },
                        "required": ["ticker", "start_date", "end_date"]
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
                            }
                        },
                        "required": ["ticker", "start_date", "end_date", "indicators"]
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
                        "required": ["ticker", "start_date", "end_date"]
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
                        "required": ["start_date", "end_date"]
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
                            "end_date": {"type": "string"}
                        },
                        "required": ["ticker", "start_date", "end_date"]
                    }
                }
            }
        ]

    def run_llm_with_tools(self, user_input: str):
        messages = [
            {"role": "system", "content": "You are a financial assistant. Choose appropriate tools for the user's request."},
            {"role": "user", "content": user_input}
        ]

        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            tools=self.tools,
            tool_choice="auto"
        )

        while response.choices[0].finish_reason == "tool_calls":
            tool_messages = []

            for tool_call in response.choices[0].message.tool_calls:
                fn_name = tool_call.function.name
                args = json.loads(tool_call.function.arguments)
                result_str = self.dispatch_function(fn_name, args)
                tool_messages.append({
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": fn_name,
                    "content": result_str or "Done."
                })

            messages.append(response.choices[0].message)
            messages.extend(tool_messages)
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                tools=self.tools,
                tool_choice="auto"
            )

    def dispatch_function(self, fn_name: str, args: dict):
        if fn_name == "plot_stock_chart":
            self.handle_plot(**args)
            return "Chart generated and saved."
        elif fn_name == "train_model":
            self.handle_train(**args)
            return "Training complete."
        elif fn_name == "analyze_stock":
            result, _ = self.handle_analysis(**args)
            return json.dumps(result, indent=2)
        elif fn_name == "get_fear_greed":
            return self.handle_fear_greed(**args).tail(3).to_string(index=False)
        elif fn_name == "summarize_market_view":
            print("check")
            return self.handle_summary(**args)
        else:
            return f"Unknown function: {fn_name}"

    def handle_plot(self, ticker: str, start_date: str, end_date: str):
        fetcher = DataFetchAgent(start=start_date, end=end_date)
        df = fetcher.fetch(ticker)
        _, enriched_df = self.analysis_agent.analyze(df)
        self.viz_agent.plot(enriched_df, ticker)

    def handle_train(self, ticker: str, start_date: str, end_date: str, indicators: List[str]):
        fetcher = DataFetchAgent(start=start_date, end=end_date)
        df = fetcher.fetch(ticker)
        ml_agent = MLAgent(feature_list=indicators)
        _, enriched_df = self.analysis_agent.analyze(df)
        acc = ml_agent.train(enriched_df)
        print(f"ML model trained on {ticker} with indicators {indicators}. Accuracy: {acc:.2f}")

    def handle_analysis(self, ticker: str, start_date: str, end_date: str):
        fetcher = DataFetchAgent(start=start_date, end=end_date)
        df = fetcher.fetch(ticker)
        fg_df = self.handle_fear_greed(start_date, end_date)

        analysis_results, enriched_df = self.analysis_agent.analyze(df)
        enriched_df = pd.merge(enriched_df, fg_df, on="date", how="left")
        return analysis_results, enriched_df

    def handle_fear_greed(self, start_date: str, end_date: str):
        return get_fear_greed_score(start_date, end_date)

    def handle_summary(self, ticker: str, start_date: str, end_date: str):
        analysis_results, enriched_df = self.handle_analysis(ticker, start_date, end_date)
        ml_accuracy = self.ml_agent.train(enriched_df)
        today_row = enriched_df.iloc[-1]
        ml_prediction = self.ml_agent.predict(today_row)
        fg_value = today_row.get("FG", 0)

        return self.generate_summary_with_llm(
            analysis_results, ml_prediction,
            f"Market summary for {ticker}",
            fg_value, ml_accuracy
        )

    def generate_summary_with_llm(self, analysis_results: dict, ml_prediction: int, user_input: str, fg_score: float, ml_accuracy: float) -> str:
        system_prompt = (
            "You are a financial analyst assistant. Based on technical indicators and model prediction (give less weight on ml prediction if its accuracy is low), "
            "you are given this instruction below to identify overbought: if 2day-RSI is greater than 90, oversold if 2day-RSI is less than 10. "
            "Summarize the market situation and must suggest a final action.\n"
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
            f"\nThe machine learning model's predicted probability of UP: {ml_prediction:.2f}\n"
            f"\nPlease write a natural language summary and suggest a final action."
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

if __name__ == "__main__":
    print("=== Tool Calling Agent ===")
    user_input = input("Enter your request (e.g., 'Show me a chart of AAPL from 2023-01-01 to 2024-01-01'): ").strip()
    analysis_agent = AnalysisAgent()
    ml_agent = MLAgent(feature_list=["close", "volume", "rsi2", "macd", "macd_signal", "sma200"])
    coordinator = ToolCallingAgent(analysis_agent, ml_agent)
    coordinator.run_llm_with_tools(user_input)
