# 5_export_for_web.py (Corrected)

import pandas as pd
import json
import os

def export_data_for_web():
    """
    Consolidates all necessary data into a single data.json file
    for the static website to use.
    """
    print("Starting data export for web...")

    try:
        predictions_df = pd.read_csv("latest_predictions.csv")
        summary_df = pd.read_csv("backtest_summary.csv")
    except FileNotFoundError as e:
        print(f"Error: Missing a required CSV file. {e}")
        print("Please run '5_automated_backtest.py' and '6_generate_predictions.py' first.")
        return

    predictions_dict = predictions_df.to_dict(orient='records')
    summary_dict = summary_df.set_index('Ticker').to_dict(orient='index')

    price_history = {}
    stock_files = os.listdir('stock_data')
    for ticker_file in stock_files:
        if ticker_file.endswith('.csv'):
            ticker = ticker_file.split('.')[0]
            df = pd.read_csv(f"stock_data/{ticker_file}")
            
            df['Date'] = pd.to_datetime(df['Date'])
            df = df[df['Date'] > (pd.Timestamp.now() - pd.DateOffset(years=1))]
            
            # --- THE FIX IS HERE ---
            # Convert the 'Date' column from Timestamp objects to simple strings
            # in the format 'YYYY-MM-DD'.
            df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
            
            price_history[ticker] = df.to_dict(orient='records')

    final_data = {
        "predictions": predictions_dict,
        "summary": summary_dict,
        "priceHistory": price_history
    }

    with open('data.json', 'w') as f:
        json.dump(final_data, f, indent=2)

    print("\nSuccessfully exported all data to 'data.json'.")
    print("This file is now ready to be deployed with your static website.")

if __name__ == "__main__":
    export_data_for_web()