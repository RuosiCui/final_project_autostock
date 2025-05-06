import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import os
from data_fetch_agent import DataFetchAgent


class VisualizationAgent:
    def __init__(self):
        # No specific initialization needed for Plotly
        pass

    def plot(self, df: pd.DataFrame, ticker: str = "Ticker", add_benchmarks: bool = True):
        if df.empty:
            print("No data to plot.")
            return

        df = df.copy()
        
        # Make sure date is in datetime format
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            # Keep date as a column for plotly
            date_col = df['date']
            df.set_index('date', inplace=True)
        else:
            # If already indexed by date
            date_col = df.index
            df.index = pd.to_datetime(df.index)

        # Calculate moving averages
        if 'close' in df.columns:
            print("Calculating moving averages...")
            df['MA20'] = df['close'].rolling(window=20).mean()
            df['MA50'] = df['close'].rolling(window=50).mean()
            df['MA200'] = df['close'].rolling(window=200).mean()
        
        # Create a figure with subplots for price, RSI, and MACD
        fig = make_subplots(
            rows=3,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=("Price Chart", "RSI", "MACD"),
            row_heights=[0.6, 0.2, 0.2]
        )
        
        # Print debug info
        print(f"Data columns: {df.columns.tolist()}")
        print(f"First few rows of data:\n{df[['close']].head() if 'close' in df.columns else 'No close column'}")
        
        # Add a line chart for the close price with high visibility
        # This is the only trace visible by default
        fig.add_trace(
            go.Scatter(
                x=date_col,
                y=df['close'],
                name='Close Price',
                line=dict(width=3, color='#2196F3'),
                mode='lines',
                hovertemplate='%{y:.2f}<extra></extra>',
                visible=True
            ),
            row=1, col=1
        )
        
        # Add 20-day MA (hidden by default)
        fig.add_trace(
            go.Scatter(
                x=date_col,
                y=df['MA20'],
                name='20-day MA',
                line=dict(width=2, color='#FFA000'),
                mode='lines',
                hovertemplate='%{y:.2f}<extra></extra>',
                visible='legendonly'
            ),
            row=1, col=1
        )
        
        # Add 50-day MA (hidden by default)
        fig.add_trace(
            go.Scatter(
                x=date_col,
                y=df['MA50'],
                name='50-day MA',
                line=dict(width=2, color='#9C27B0'),
                mode='lines',
                hovertemplate='%{y:.2f}<extra></extra>',
                visible='legendonly'
            ),
            row=1, col=1
        )
        
        # Add 200-day MA if we have enough data (hidden by default)
        if len(df) >= 200:
            fig.add_trace(
                go.Scatter(
                    x=date_col,
                    y=df['MA200'],
                    name='200-day MA',
                    line=dict(width=2, color='#F44336'),
                    mode='lines',
                    hovertemplate='%{y:.2f}<extra></extra>',
                    visible='legendonly'
                ),
                row=1, col=1
            )
            
        # Add benchmark tickers (SPY and QQQ)
        if add_benchmarks:
            try:
                print("Adding benchmark tickers (SPY and QQQ)...")
                # Get the date range from the current dataframe
                start_date = df.index[0] if isinstance(df.index, pd.DatetimeIndex) else df['date'].min()
                end_date = df.index[-1] if isinstance(df.index, pd.DatetimeIndex) else df['date'].max()
                
                # Format dates as strings for the DataFetchAgent
                start_str = pd.to_datetime(start_date).strftime('%Y-%m-%d')
                end_str = pd.to_datetime(end_date).strftime('%Y-%m-%d')
                
                # Create a DataFetchAgent
                fetcher = DataFetchAgent(start=start_str, end=end_str)
                
                # Fetch SPY data
                spy_df = fetcher.fetch('SPY')
                if not spy_df.empty:
                    # Normalize SPY data to match the scale of the main ticker
                    first_close = df['close'].iloc[0]
                    first_spy = spy_df['close'].iloc[0]
                    spy_df['normalized'] = spy_df['close'] * (first_close / first_spy)
                    
                    fig.add_trace(
                        go.Scatter(
                            x=spy_df['date'],
                            y=spy_df['normalized'],
                            name='SPY (normalized)',
                            line=dict(width=2, color='rgba(0, 128, 0, 0.7)', dash='dash'),
                            mode='lines',
                            hovertemplate='%{y:.2f}<extra></extra>',
                            visible='legendonly'
                        ),
                        row=1, col=1
                    )
                
                # Fetch QQQ data
                qqq_df = fetcher.fetch('QQQ')
                if not qqq_df.empty:
                    # Normalize QQQ data to match the scale of the main ticker
                    first_qqq = qqq_df['close'].iloc[0]
                    qqq_df['normalized'] = qqq_df['close'] * (first_close / first_qqq)
                    
                    fig.add_trace(
                        go.Scatter(
                            x=qqq_df['date'],
                            y=qqq_df['normalized'],
                            name='QQQ (normalized)',
                            line=dict(width=2, color='rgba(128, 0, 128, 0.7)', dash='dash'),
                            mode='lines',
                            hovertemplate='%{y:.2f}<extra></extra>',
                            visible='legendonly'
                        ),
                        row=1, col=1
                    )
            except Exception as e:
                print(f"Error adding benchmark tickers: {e}")
                
        # Add RSI charts (hidden by default)
        if 'rsi2' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=date_col,
                    y=df['rsi2'],
                    name='RSI (2)',
                    line=dict(width=2, color='#FF9800'),
                    mode='lines',
                    hovertemplate='%{y:.2f}<extra></extra>',
                    visible='legendonly'
                ),
                row=2, col=1
            )
            
        if 'rsi14' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=date_col,
                    y=df['rsi14'],
                    name='RSI (14)',
                    line=dict(width=2, color='#E91E63'),
                    mode='lines',
                    hovertemplate='%{y:.2f}<extra></extra>',
                    visible='legendonly'
                ),
                row=2, col=1
            )
            
        # Add overbought/oversold lines for RSI
        fig.add_shape(
            type="line", line=dict(dash='dash', width=1, color='red'),
            y0=70, y1=70, x0=date_col.iloc[0], x1=date_col.iloc[-1],
            row=2, col=1
        )
        fig.add_shape(
            type="line", line=dict(dash='dash', width=1, color='green'),
            y0=30, y1=30, x0=date_col.iloc[0], x1=date_col.iloc[-1],
            row=2, col=1
        )
        
        # Set y-axis range for RSI
        fig.update_yaxes(range=[0, 100], row=2, col=1)
        
        # Add MACD chart (hidden by default)
        if 'macd' in df.columns and 'macd_signal' in df.columns:
            # MACD line
            fig.add_trace(
                go.Scatter(
                    x=date_col,
                    y=df['macd'],
                    name='MACD',
                    line=dict(width=2, color='#2196F3'),
                    mode='lines',
                    hovertemplate='%{y:.2f}<extra></extra>',
                    visible='legendonly'
                ),
                row=3, col=1
            )
            
            # Signal line
            fig.add_trace(
                go.Scatter(
                    x=date_col,
                    y=df['macd_signal'],
                    name='Signal',
                    line=dict(width=2, color='#FF9800', dash='dash'),
                    mode='lines',
                    hovertemplate='%{y:.2f}<extra></extra>',
                    visible='legendonly'
                ),
                row=3, col=1
            )
            
            # Calculate MACD histogram if not in dataframe
            if 'macd_hist' not in df.columns:
                df['macd_hist'] = df['macd'] - df['macd_signal']
            
            # MACD histogram
            colors = ['green' if val >= 0 else 'red' for val in df['macd_hist']]
            fig.add_trace(
                go.Bar(
                    x=date_col,
                    y=df['macd_hist'],
                    name='Histogram',
                    marker=dict(color=colors),
                    hovertemplate='%{y:.2f}<extra></extra>',
                    visible='legendonly'
                ),
                row=3, col=1
            )
            
            # Add zero line
            fig.add_shape(
                type="line", line=dict(width=1, color='gray'),
                y0=0, y1=0, x0=date_col.iloc[0], x1=date_col.iloc[-1],
                row=3, col=1
            )
        
        # Update layout for the multi-panel chart
        fig.update_layout(
            # Move title to the top left and adjust its position
            title=dict(
                text=f"{ticker} Price Chart",
                x=0.01,
                y=0.98,
                xanchor='left',
                yanchor='top',
                font=dict(size=20)
            ),
            height=900,  # Increased height to give more space between panels
            template="plotly_dark",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                bgcolor='rgba(50, 50, 50, 0.7)',  # Semi-transparent background
                bordercolor='rgba(255, 255, 255, 0.3)',
                borderwidth=1
            ),
            xaxis_rangeslider_visible=False,
            hovermode="x",
            plot_bgcolor='rgba(30, 30, 30, 1)',
            paper_bgcolor='rgba(30, 30, 30, 1)',
            margin=dict(t=80, b=50, l=50, r=50)  # Increased top margin for title and buttons
        )
        
        # Update y-axis titles and increase spacing between panels
        fig.update_yaxes(title_text="Price", row=1, col=1, title_font=dict(size=14))
        fig.update_yaxes(title_text="RSI", row=2, col=1, title_font=dict(size=14))
        fig.update_yaxes(title_text="MACD", row=3, col=1, title_font=dict(size=14))
        
        # No need for additional margin adjustment as it's included in the main layout
        
        # Add grid lines for better readability
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(80, 80, 80, 0.3)')
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(80, 80, 80, 0.3)')
        
        # Add range selector for time periods with improved visibility and positioning
        fig.update_xaxes(
            rangeslider_visible=False,
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=3, label="3m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(step="all")
                ]),
                bgcolor='rgba(50, 50, 50, 0.8)',  # Darker background for buttons
                font=dict(color='#FFEB3B'),  # Yellow text for better visibility
                activecolor='#4CAF50',  # Green for active button
                x=0.5,  # Center position
                y=1.06,  # Higher position to avoid overlap with title
                borderwidth=1,
                bordercolor='#FFEB3B'  # Yellow border
            )
        )
        
        # Save the figure to an HTML file instead of showing it directly
        print("Saving chart to 'stock_chart.html'...")
        fig.write_html("stock_chart.html")
        print(f"Chart saved to: {os.path.abspath('stock_chart.html')}")
        print("Please open this file in your browser to view the chart.")
        
        # Also try to show the figure (as a backup)
        try:
            fig.show()
        except Exception as e:
            print(f"Could not display chart directly: {e}")
