import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import threading


class VisualizationAgent:
    def __init__(self):
        plt.style.use('seaborn-v0_8-darkgrid')

    def plot(self, df: pd.DataFrame, ticker: str = "Ticker"):
        if df.empty:
            print("No data to plot.")
            return

        df = df.copy()
        df.set_index('date', inplace=True)

        fig, (ax_price, ax_rsi, ax_macd) = plt.subplots(3, 1, figsize=(14, 10), sharex=True,
                                                        gridspec_kw={'height_ratios': [3, 1, 1]})
        fig.suptitle(f"{ticker} Price and Indicators", fontsize=16)

        # 1️⃣ Price + Volume
        ax_price.plot(df.index, df['close'], label='Close', linewidth=1.8)

        if 'close' in df.columns:
            df['MA20'] = df['close'].rolling(20).mean()
            df['MA50'] = df['close'].rolling(50).mean()
            df['MA200'] = df['close'].rolling(200).mean()

            ax_price.plot(df.index, df['MA20'], label='20d MA', linestyle='--', linewidth=1.2)
            ax_price.plot(df.index, df['MA50'], label='50d MA', linestyle='--', linewidth=1.2)
            ax_price.plot(df.index, df['MA200'], label='200d MA', linestyle='--', linewidth=1.2)

        ax_price.set_ylabel("Price")
        ax_price.legend(loc='upper left')

        ax_price2 = ax_price.twinx()
        ax_price2.bar(df.index, df['volume'], color='gray', alpha=0.3, label='Volume')
        ax_price2.set_ylabel("Volume")
        ax_price2.legend(loc='upper right')

        # 2️⃣ RSI
        if 'rsi14' in df.columns:
            ax_rsi.plot(df.index, df['rsi14'], color='purple', label='RSI (14)', linewidth=1.5)
            ax_rsi.axhline(70, color='red', linestyle='--', linewidth=1)
            ax_rsi.axhline(30, color='green', linestyle='--', linewidth=1)
            ax_rsi.set_ylim(0, 100)
            ax_rsi.set_ylabel("RSI")
            ax_rsi.legend(loc='upper left')

        # 3️⃣ MACD
        if 'macd' in df.columns and 'macd_signal' in df.columns:
            ax_macd.plot(df.index, df['macd'], label='MACD', color='blue', linewidth=1.2)
            ax_macd.plot(df.index, df['macd_signal'], label='MACD Signal', color='orange', linestyle='--')
            ax_macd.axhline(0, color='black', linewidth=0.5)
            ax_macd.set_ylabel("MACD")
            ax_macd.legend(loc='upper left')

        # 📅 Format x-axis
        ax_macd.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        fig.autofmt_xdate()

        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        plt.show()

    
