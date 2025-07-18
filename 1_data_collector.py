# 1_data_collector.py

import yfinance as yf
import pandas as pd
import os

# --- Settings ---
TICKERS = [
    ### TIER 1: ADD MARKET BENCHMARK ###
    "SPY",   # S&P 500 ETF - Market Benchmark
    # --- Chosen Tickers ---
    "GOOGL", # Google
    "MSFT",  # Microsoft
    "META",  # Meta Platforms
    "AMZN",  # Amazon
    "NVDA",  # Nvidia
    "INTC",  # Intel
    "AMD",   # Advanced Micro Devices

    # # --- Top Tech & Cloud ---
    # "TSLA",  # Tesla
    # "AAPL",  # Apple
    # "CRM",   # Salesforce
    # "ORCL",  # Oracle
    # "ADBE",  # Adobe

    # # --- Semiconductors ---
    # "TSM",   # Taiwan Semiconductor
    # "AVGO",  # Broadcom
    # "QCOM",  # Qualcomm
    # "TXN",   # Texas Instruments

    # # --- Top Finance & Banking ---
    # "JPM",   # JPMorgan Chase
    # "V",     # Visa
    # "MA",    # Mastercard
    # "BAC",   # Bank of America
    # "GS",    # Goldman Sachs
    # "MS",    # Morgan Stanley
    # "BLK",   # BlackRock
]
PERIOD = "5y" 
DATA_DIR = "stock_data"

# --- Code ---
def download_stock_data():
    """Downloads data, flattens MultiIndex if present, and saves to CSV."""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"Created directory: {DATA_DIR}")

    for ticker in TICKERS:
        try:
            print(f"Downloading data for {ticker}...")
            # We are ignoring the FutureWarning as it's informational
            data = yf.download(ticker, period=PERIOD, progress=False, auto_adjust=True)
            
            # THE CRITICAL ONE-LINE FIX:
            # Check if the columns are a MultiIndex and flatten it.
            if isinstance(data.columns, pd.MultiIndex):
                print("MultiIndex detected. Flattening columns...")
                # This line takes the top-level headers (Close, High, etc.)
                # and makes them the new, simple column names.
                data.columns = data.columns.get_level_values(0)

            if data.empty:
                print(f"No data found for {ticker}, skipping.")
                continue

            file_path = os.path.join(DATA_DIR, f"{ticker}.csv")
            data.to_csv(file_path, index=True, index_label="Date")
            
            print(f"Successfully saved data for {ticker} to {file_path}")

        except Exception as e:
            print(f"Could not download data for {ticker}. Error: {e}")

if __name__ == "__main__":
    download_stock_data()