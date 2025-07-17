## Activate the venv
.\.venv\Scripts\Activate.ps1

## Update the data.json
python main.py

================================================================

# AI Quant Desk

This project is an end-to-end algorithmic trading pipeline. It automates the process of fetching stock data, training a machine learning model for each stock to predict price movements, backtesting the strategy's performance, and generating real-time trading signals. All results and signals are consolidated and displayed on a clean, interactive web dashboard.

## Features

-   **Automated Data Pipeline:** A single command (`python main.py`) runs the entire workflow from data collection to final output.
-   **Data Collection:** Downloads 5 years of historical stock data for a predefined list of tech and finance tickers using `yfinance`.
-   **Machine Learning Model Training:** Trains a unique Random Forest Classifier for each stock to predict whether the price will increase by a set threshold within a future period.
-   **Feature Engineering:** Automatically calculates technical indicators (RSI, MACD, Bollinger Bands, ATR) using `pandas-ta` to serve as features for the models.
-   **Automated Backtesting:** Uses the `backtesting.py` library to evaluate the performance of the trained models on historical data, generating key metrics like Sharpe Ratio, Win Rate, and Total Return.
-   **Signal Generation:** Creates up-to-date 'BUY' or 'HOLD/SELL' signals with confidence scores for each stock.
-   **Web Dashboard:** An interactive, single-page dashboard (`index.html`) built with vanilla JavaScript and Plotly.js to visualize predictions, backtest summaries, and price history.

## Technology Stack

-   **Backend & ML:** Python
-   **Core Libraries:** `pandas`, `scikit-learn`, `yfinance`, `pandas-ta`, `joblib`, `backtesting.py`
-   **Frontend:** HTML, CSS, JavaScript
-   **Charting:** Plotly.js

## How to Run

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/your-username/ai-quant-desk.git
    cd ai-quant-desk
    ```

2.  **Set up a Virtual Environment (Recommended)**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install Dependencies**
    A `requirements.txt` file can be created with the following content:
    ```
    yfinance
    pandas
    pandas_ta
    scikit-learn
    joblib
    backtesting
    ```
    Then, install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Full Pipeline**
    This command will execute all the Python scripts in order, from data collection to the final JSON export.
    ```bash
    python main.py
    ```

5.  **View the Dashboard**
    Open the `index.html` file in your favorite web browser to see the results.

## How It Works: The Pipeline

The project is orchestrated by `main.py`, which runs the following scripts in sequence:

1.  **`1_data_collector.py`**: Fetches 5 years of daily stock data for the tickers defined in the script. It saves the data for each stock into a separate CSV file inside the `stock_data/` directory.

2.  **`2_model_trainer.py`**: For each stock data CSV, this script:
    -   Calculates technical analysis features (RSI, MACD, Bollinger Bands, ATR).
    -   Defines a target variable: `1` if the stock price increases by more than 2% within the next 5 days, and `0` otherwise.
    -   Splits the data chronologically into a training set (80%) and a testing set (20%).
    -   Trains a `RandomForestClassifier` model on the training data.
    -   Saves the trained model as a `.pkl` file in the `models/` directory.

3.  **`3_automated_backtest.py`**:
    -   Loads each trained model and its corresponding historical data.
    -   Uses the `backtesting.py` library to run a simulation where the model's predictions (1=buy, 0=sell) dictate trading actions.
    -   Aggregates the performance stats for all tickers and saves them to `backtest_summary.csv`.

4.  **`4_generate_predictions.py`**:
    -   For each stock, it fetches the latest data and calculates the same technical features.
    -   It loads the corresponding trained model to predict the next trading signal ('BUY' or 'HOLD/SELL').
    -   It also calculates the model's confidence in that prediction.
    -   The results are saved to `latest_predictions.csv`.

5.  **`5_export_for_web.py`**:
    -   This final script acts as a data consolidator for the frontend.
    -   It reads `latest_predictions.csv` and `backtest_summary.csv`.
    -   It also reads the last 1-year of price history for each stock.
    -   All this information is combined into a single, structured `data.json` file, which is what the dashboard consumes.

6.  **`index.html`**:
    -   This is the static web page that serves as the dashboard.
    -   When opened, its JavaScript code fetches `data.json`.
    -   It dynamically populates the ticker grid, performance tables, and interactive price charts using the data from the JSON file.

## Disclaimer

This project is for educational purposes only and should not be considered financial advice. The trading signals generated are based on a simple model and historical data, which is not indicative of future results. Do not risk money that you are not prepared to lose.