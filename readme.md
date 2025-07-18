## Activate the venv
.\.venv\Scripts\Activate.ps1

## Update the data.json
python main.py

================================================================

# AI Quant Desk

This project is an advanced, end-to-end algorithmic trading pipeline. It automates data fetching, robust model training, and realistic backtesting to generate predictive trading signals. The entire system is designed to find a statistical edge in the market by leveraging sophisticated machine learning techniques and comprehensive feature engineering. All results are consolidated and displayed on a clean, interactive web dashboard.

---

## Quick Start

1.  **Set up the environment and install dependencies:**
    ```bash
    # Create and activate a virtual environment (recommended)
    python -m venv .venv
    .\.venv\Scripts\Activate.ps1

    # Install all required libraries
    pip install -r requirements.txt
    ```

2.  **Run the entire data pipeline:**
    ```bash
    python main.py
    ```

3.  **View the results:**
    *   Open `index.html` in your browser.
    *   Check `latest_predictions.csv` for the most recent signals.
    *   Check `backtest_summary.csv` for detailed performance metrics.

## Key Features

-   **Automated Pipeline:** A single command (`python main.py`) runs the entire workflow, from data collection to the final `data.json` for the web dashboard.
-   **Advanced Feature Engineering:** Calculates a wide array of technical indicators (RSI, MACD, etc.) and incorporates **market context** using S&P 500 data. It also includes advanced metrics like **trend strength filters** (e.g., SMA 50 vs. 200), **mean-reversion indicators** (distance from moving average), and **second-order indicators** (RSI volatility).
-   **Robust ML Modeling:**
    -   Uses **LightGBM**, a powerful and efficient gradient boosting framework, for classification.
    -   Performs **hyperparameter tuning** for each model using `GridSearchCV`.
    -   Employs **Time-Series Cross-Validation** (`TimeSeriesSplit`) to find the best models in a chronologically-aware manner, simulating real-world performance far more accurately than a standard train/test split.
-   **Adaptive Target Labeling:** Instead of a fixed percentage, it uses a **quantile-based system** to define targets. The top 20% of future returns are labeled 'Buy', and the bottom 20% are labeled 'Sell', making the system adaptive to each stock's unique volatility.
-   **Handles Class Imbalance:** Intelligently manages the natural imbalance of Buy/Sell/Hold signals by using the `class_weight='balanced'` parameter, forcing the model to pay attention to the rare but critical trading opportunities.
-   **Explainable AI (XAI):** The dashboard displays the top "Model Drivers" (feature importances) for each stock, providing insight into *why* a model is making a particular decision.
-   **Comprehensive Backtesting:** Uses the `backtesting.py` library to simulate strategy performance, generating key metrics like Sharpe Ratio, Max Drawdown, and Win Rate. The simulation includes risk management via a configurable **stop-loss**.
-   **Interactive Dashboard:** A sleek, modern dashboard (`index.html`) built with vanilla JavaScript and Plotly.js. It features a rich **glossary** with tooltips explaining every metric and indicator, making complex data accessible to all users.

## Technology Stack

-   **Backend & ML:** Python
-   **Core Libraries:**
    -   `pandas`: Data manipulation and analysis.
    -   `lightgbm` & `scikit-learn`: For building and evaluating machine learning models.
    -   `yfinance`: Downloading historical stock data from Yahoo! Finance.
    -   `pandas-ta`: Calculating technical analysis indicators.
    -   `joblib`: Saving and loading trained model objects.
    -   `backtesting.py`: For running historical strategy simulations.
-   **Frontend:** HTML, CSS, JavaScript (no frameworks)
-   **Charting:** Plotly.js

## How to Run

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/your-username/ai-quant-desk.git
    cd ai-quant-desk
    ```

2.  **Set up a Virtual Environment (Recommended)**
    ```bash
    python -m venv .venv
    # On Windows PowerShell
    .\.venv\Scripts\Activate.ps1
    # On macOS/Linux
    # source .venv/bin/activate
    ```

3.  **Install Dependencies**
    The `requirements.txt` file contains all necessary libraries.
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Full Pipeline**
    This command executes all the Python scripts in order, from data collection to the final JSON export.
    ```bash
    python main.py
    ```

5.  **View the Dashboard**
    Simply open the `index.html` file in your favorite web browser.

## The Pipeline Explained

The project is orchestrated by `main.py`, which runs the following scripts in sequence:

1.  **`1_data_collector.py`**: Fetches 5 years of daily stock data for the tickers defined in the script. It saves the data for each stock into a separate CSV file inside the `stock_data/` directory and handles potential data formatting issues from the API.

2.  **`2_model_trainer.py`**: This is the core of the AI system. For each stock, it:
    -   Calculates a comprehensive set of technical and market-based features.
    -   Defines a target variable using **quantile-based labeling**: 'Buy' (2) for top-tier future returns, 'Sell' (0) for bottom-tier, and 'Hold' (1) for everything in between.
    -   Splits the data chronologically into a training set (80%) and a final hold-out test set (20%).
    -   Uses `GridSearchCV` with `TimeSeriesSplit` to find the best hyperparameters for a `LightGBM` model, training it with `class_weight='balanced'`.
    -   Evaluates the best model on the unseen test set.
    -   Saves a model "payload" (`.pkl`) containing the trained model, the exact order of features used, and their importances.

3.  **`3_automated_backtest.py`**:
    -   Loads each trained model payload.
    -   Runs a simulation on the historical data, using the model's predictions (0, 1, 2) to execute trades (Sell, Hold, Buy). Includes a **stop-loss for risk management**.
    -   Aggregates the performance stats for all tickers and saves them to `backtest_summary.csv`.

4.  **`4_generate_predictions.py`**:
    -   For each stock, it loads the latest data and calculates all necessary features.
    -   It loads the corresponding model payload and uses it to predict the next trading signal, ensuring the feature order matches training precisely.
    -   The final predictions and confidence scores are saved to `latest_predictions.csv`.

5.  **`5_export_for_web.py`**:
    -   Consolidates all the necessary data for the frontend.
    -   It reads `latest_predictions.csv`, `backtest_summary.csv`, the model feature importances, and the last year of price history for each stock.
    -   All this information is combined into a single, structured `data.json` file.

6.  **`index.html`**:
    -   A static web page that acts as the dashboard.
    -   Its JavaScript fetches the `data.json` file on load.
    -   It dynamically populates the ticker grid, performance tables, model driver charts, interactive price charts, and **educational tooltips**.

## Disclaimer

This project is for educational purposes only and should not be considered financial advice. The trading signals generated are based on historical data, which is not indicative of future results. Financial markets are inherently unpredictable. Do not risk money that you are not prepared to lose.