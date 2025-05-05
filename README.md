# AutoStock: Interactive Financial Analysis System

## Overview

AutoStock is a comprehensive financial analysis system that combines multiple specialized agents to provide stock market analysis, visualization, prediction, and backtesting capabilities. The system leverages large language models (LLMs) and machine learning to offer insights and recommendations for stock trading decisions.

## Key Features

- **Interactive Natural Language Interface**: Ask questions and request analyses in plain English
- **Technical Indicator Analysis**: Calculate and interpret key technical indicators
- **Stock Visualization**: Generate interactive charts with technical indicators
- **Machine Learning Predictions**: Train models on custom indicator combinations
- **Market Sentiment Analysis**: Incorporate CNN Fear & Greed Index data
- **Comprehensive Backtesting**: Validate strategies on historical data
- **Classification Metrics**: Evaluate strategy performance with precision, recall, and F1 score

## System Architecture

The system follows a multi-agent architecture where specialized agents handle different aspects of financial analysis:

### Core Components

#### 1. Interactive Tool Calling Agent (`interactive_tool_calling_agent.py`)

The central coordinator that:
- Processes natural language requests
- Routes tasks to appropriate specialized agents
- Integrates results into coherent responses
- Provides tools for various financial analysis tasks

Available tools:
- `plot_stock_chart`: Generate interactive stock charts with technical indicators
- `train_model`: Train ML models on custom indicator combinations
- `analyze_stock`: Run technical analysis on historical stock data
- `get_fear_greed`: Retrieve CNN Fear & Greed index data
- `summarize_market_view`: Generate buy/sell recommendations
- `run_backtest`: Perform comprehensive strategy backtesting

#### 2. Analysis Agent (`analysis_agent.py`)

Responsible for technical analysis:
- Calculates technical indicators (RSI, MACD, Bollinger Bands, etc.)
- Identifies support and resistance levels
- Detects moving average crossovers
- Analyzes price trends and patterns

#### 3. ML Agent (`ml_agent.py`)

Handles machine learning predictions:
- Trains models on customizable indicator combinations
- Predicts price movement probabilities
- Validates model performance
- Supports feature importance analysis

#### 4. Data Fetch Agent (`data_fetch_agent.py`)

Manages data acquisition:
- Retrieves historical stock data from Yahoo Finance
- Handles date range filtering
- Ensures data quality and consistency
- Provides data preprocessing capabilities

#### 5. Fear & Greed Agent (`fear_greed_agent.py`)

Incorporates market sentiment:
- Retrieves CNN Fear & Greed Index data
- Interprets market sentiment indicators
- Provides historical sentiment context
- Enhances analysis with sentiment data

#### 6. Visualization Agent (`visualization_agent.py`)

Creates visual representations:
- Generates interactive stock charts
- Plots technical indicators
- Visualizes support and resistance levels
- Creates custom visualization layouts

#### 7. Interactive Backtest Agent (`interactive_run_backtest.py`)

Enables strategy validation:
- Runs backtests on historical data
- Allows customization of lookback windows
- Supports iterative indicator selection
- Provides comprehensive performance metrics

#### 8. Feature Resolver (`feature_resolver.py`)

Handles natural language processing:
- Maps user-friendly indicator names to canonical forms
- Extracts indicator requests from natural language
- Validates indicator availability
- Ensures consistent indicator naming

## Usage Examples

### Basic Stock Analysis

```
Enter your request: Should I buy or sell MSFT based on technical indicators?
```

### Custom Chart Generation

```
Enter your request: Show me a chart for TSLA from 2022-01-01 to 2023-01-01 with RSI and MACD
```

### Custom Model Training

```
Enter your request: Train a model for AAPL using RSI, volume ratio, and SMA50
```

### Interactive Backtesting

```
Enter your request: Run an interactive backtest of the trading strategy
```

## Performance Metrics

The system provides comprehensive performance metrics for backtesting:

- **Accuracy**: Overall percentage of correct predictions
- **Precision**: Accuracy of positive (BUY) predictions
- **Recall**: Ability to find all positive instances
- **F1 Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Detailed breakdown of prediction results

## Technical Implementation

- **Language**: Python 3.10+
- **Data Source**: Yahoo Finance API
- **ML Framework**: Scikit-learn
- **LLM Integration**: OpenAI GPT-4o
- **Visualization**: Matplotlib, Plotly
- **Natural Language Processing**: Custom feature extraction

## Future Enhancements

- Portfolio optimization recommendations
- Multi-stock correlation analysis
- Fundamental analysis integration
- Real-time market data streaming
- Automated trading capabilities
- Enhanced visualization options

## Getting Started

1. Install required dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Run the interactive agent:
   ```
   python final_project_autostock/interactive_tool_calling_agent.py
   ```

3. Enter your OpenAI API key when prompted

4. Start asking questions and analyzing stocks!