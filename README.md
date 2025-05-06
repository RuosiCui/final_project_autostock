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

Creates interactive visual representations:
- Generates interactive Plotly-based stock charts
- Displays candlestick patterns with hover information
- Plots technical indicators (RSI, MACD) with overbought/oversold levels
- Includes time period selectors (1m, 3m, 6m, 1y, YTD, All)
- Provides zoom, pan, and export capabilities
- Displays detailed information on hover

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

## Multi-Agent Workflow

The system uses a sophisticated workflow to process user requests through multiple specialized agents. Here's how the system handles different types of requests:

### 1. Request Processing Flow

```
User Request → Interactive Tool Calling Agent → Specialized Agents → Results Integration → Response
```

1. **User Input Processing**:
   - The Interactive Tool Calling Agent receives natural language input
   - It extracts potential feature names using the Feature Resolver
   - It sends the request to OpenAI's GPT-4o model for tool selection

2. **Tool Selection**:
   - GPT-4o analyzes the request and selects the appropriate tool
   - The agent dispatches the request to the corresponding function handler

3. **Date Parameter Handling**:
   - If date parameters (start_date, end_date) are not provided:
     - The system prompts the user to enter dates
     - Default values are used if the user doesn't specify (start_date = "2010-01-01", end_date = today's date)

4. **Specialized Agent Execution**:
   - The appropriate specialized agents are invoked based on the tool
   - Results are collected and formatted for the response

5. **Response Generation**:
   - Results are integrated into a coherent response
   - The response is returned to the user

### 2. Tool-Specific Workflows

#### Stock Chart Generation Workflow

```
User Request → Interactive Tool Calling Agent → Data Fetch Agent → Analysis Agent → Visualization Agent → Interactive Chart
```

1. User requests a chart (e.g., "Show me a chart for TSLA")
2. Interactive Tool Calling Agent selects the `plot_stock_chart` tool
3. If dates aren't provided, the system prompts for them or uses defaults
4. Data Fetch Agent retrieves historical stock data from Yahoo Finance
5. Analysis Agent calculates technical indicators
6. Visualization Agent creates an interactive Plotly chart
7. Chart is saved as HTML and displayed to the user

#### ML Model Training Workflow

```
User Request → Interactive Tool Calling Agent → Feature Resolver → Data Fetch Agent → Analysis Agent → ML Agent → Training Results
```

1. User requests model training (e.g., "Train a model for AAPL using RSI and volume")
2. Interactive Tool Calling Agent selects the `train_model` tool
3. Feature Resolver extracts and validates indicator names
4. If dates aren't provided, the system prompts for them or uses defaults
5. Data Fetch Agent retrieves historical stock data
6. Analysis Agent calculates technical indicators
7. ML Agent trains a logistic regression model on the specified indicators
8. Training accuracy is returned to the user

#### Market Analysis Workflow

```
User Request → Interactive Tool Calling Agent → Data Fetch Agent → Analysis Agent → Fear & Greed Agent → Analysis Results
```

1. User requests analysis (e.g., "Analyze MSFT technical indicators")
2. Interactive Tool Calling Agent selects the `analyze_stock` tool
3. If dates aren't provided, the system prompts for them or uses defaults
4. Data Fetch Agent retrieves historical stock data
5. Analysis Agent calculates technical indicators
6. Fear & Greed Agent retrieves market sentiment data
7. Results are merged and returned to the user

#### Market Summary Workflow

```
User Request → Interactive Tool Calling Agent → Data Fetch Agent → Analysis Agent → Fear & Greed Agent → ML Agent → LLM → Buy/Sell Recommendation
```

1. User requests a recommendation (e.g., "Should I buy or sell AAPL?")
2. Interactive Tool Calling Agent selects the `summarize_market_view` tool
3. If dates aren't provided, the system prompts for them or uses defaults
4. Data Fetch Agent retrieves historical stock data
5. Analysis Agent calculates technical indicators
6. Fear & Greed Agent retrieves market sentiment data
7. ML Agent trains a model and makes a prediction
8. All data is sent to GPT-4o for comprehensive analysis
9. A detailed market summary with buy/sell recommendation is returned

#### Interactive Backtesting Workflow

```
User Request → Interactive Tool Calling Agent → Interactive Backtest Agent → Data Fetch Agent → Analysis Agent → ML Agent → LLM → Backtest Results
```

1. User requests backtesting (e.g., "Run an interactive backtest")
2. Interactive Tool Calling Agent selects the `run_backtest` tool
3. Interactive Backtest Agent prompts for ticker, date range, and indicators
4. Data Fetch Agent retrieves historical stock data
5. Analysis Agent calculates technical indicators
6. ML Agent trains on initial data window
7. For each trading day in the test period:
   - ML Agent makes a prediction
   - LLM generates a market summary and recommendation
   - Results are recorded and evaluated
8. Comprehensive performance metrics are calculated and displayed
9. Results are saved to CSV files

### 3. Date Parameter Handling

The system now uses a flexible approach to date parameters:

1. **Optional Date Parameters**:
   - Date parameters (start_date, end_date) are now optional in all tool definitions
   - This improves robustness by not relying on the LLM to detect dates correctly

2. **Interactive Date Prompting**:
   - When dates aren't provided, the system prompts the user
   - Clear prompts show default values (start_date = "2010-01-01", end_date = today's date)

3. **Default Values**:
   - If the user doesn't provide dates when prompted, sensible defaults are used
   - This ensures the system always has a valid date range to work with

4. **Date Validation**:
   - The system validates and formats dates consistently
   - This prevents errors from invalid date formats or ranges

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
- **Visualization**: Plotly (interactive charts)
- **Natural Language Processing**: Custom feature extraction

## Future Enhancements

- Portfolio optimization recommendations
- Multi-stock correlation analysis
- Fundamental analysis integration
- Real-time market data streaming
- Automated trading capabilities
- Additional technical indicators
- Pattern recognition visualization
- Correlation heatmaps

## Getting Started

1. Install required dependencies:
   ```
   pip install -r requirements.txt
   pip install -r requirements-viz.txt
   ```

2. Test the visualization (optional):
   ```
   python final_project_autostock/test_visualization.py
   ```
   
   ## System Architecture Diagram
   
   Below is a diagram showing how the different agents in the AutoStock system interact with each other:
   
   ```
                        +----------------+
                        |      User      |<--------------------------------------------+
                        +-------+--------+                                             |
                              |^                                                       |
                              || Natural Language Requests                             |
                              v|                                                       |
                  +---------+-----------------------+                                  |
                  |                                 |                                  |
                  | Interactive Tool Calling Agent  |                                  |
                  |     (Central Coordinator)       |                                  |
                  |                                 |                                  |
                  +--+--------+---------------------+                                  |
                  |^             ^|              |^            \ ^                     |
                  ||             ||              ||             \ \                    |
                  ||             ||              ||              \ \                   |
                  v|             |v              v|               v \                  |
 +------------------+   +---------------+    +---------------+    +------------+       |
 | Interactive      |   | Visual Agent  |    | Technical     |    | ML Agent   |       |
 | Backtest Agent   |   |               |    | Analysis Agent|  + |            |       |
 +--------+---------+   +-------^-------+    +------^--------+    +------^-----+       |
          |                      |                   |      |           |              |
          |                      |                   |      |           |              |
          |                      |                   |      |           |              |
          |              +-------+---------+         |      |           |              |
          |              |                 |         |      |           |              |
          |              | Data Fetch Agent|<--------+      |           |              |
          |              | (Yahoo Finance) |                |           |              |
          |              |                 |                |           |              |
          |              +-----------------+                |           |              |
          |                                                 |           |              |
          |                                                 |           |              |
          |                                                 |           |              |
          |             +--------+----------+               |           |              |
          |             |                   |<--------------+           |              |
          |             | Final LLM         |<--------------------------+              |
          +------------>| Response          |------------------------------------------+
                        |                   |            
                        +--------+----------+                    
                                    
    
   ```
   
   ### Data Flow Explanation
   
   1. **User → Interactive Tool Calling Agent**: User submits natural language requests
   
   2. **Interactive Tool Calling Agent → Analysis Agent**: Sends requests for technical indicator analysis
   
   3. **Interactive Tool Calling Agent → Visual Agent**: Sends requests for chart generation
   
   4. **Interactive Tool Calling Agent → ML Agent**: Sends requests for predictions
   
   5. **Analysis Agent → Data Fetch Agent**: Gets historical stock data for technical analysis
   
   6. **Visual Agent → Data Fetch Agent**: Gets stock data for visualization
   
   7. **ML Agent → Analysis Agent**: Uses technical indicators for training and prediction
   
   8. **Interactive Backtest Agent → Final LLM**: Sends backtest requests to the LLM
   
   9. **ML Agent → Final LLM**: Provides prediction results to the LLM
   
   10. **Analysis Agent → Final LLM**: Provides technical indicator results to the LLM
   
   11. **Final LLM → User**: Delivers final predictions and recommendations
   
   This architecture enables the system to process complex financial analysis tasks by breaking them down into specialized components that work together seamlessly.

3. Run the interactive agent:
   ```
   python final_project_autostock/interactive_tool_calling_agent.py
   ```

4. Enter your OpenAI API key when prompted

5. Start asking questions and analyzing stocks!

## Visualization Enhancements

The visualization system has been upgraded from static matplotlib charts to interactive Plotly charts with the following features:

- **Candlestick Charts**: View OHLC (Open, High, Low, Close) data in traditional candlestick format
- **Interactive Elements**: Zoom, pan, and hover for detailed information
- **Time Period Selection**: Easily switch between different time ranges (1m, 3m, 6m, 1y, YTD, All)
- **Enhanced Indicators**: Clearer visualization of RSI, MACD with overbought/oversold levels
- **Hover Details**: Get precise values when hovering over any data point
- **Dark Theme**: Modern dark theme for better contrast and reduced eye strain
- **MACD Histogram**: Color-coded histogram showing bullish/bearish momentum

To test the visualization independently:
```
python final_project_autostock/test_visualization.py