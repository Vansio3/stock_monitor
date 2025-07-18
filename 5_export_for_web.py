# 5_export_for_web.py (UPDATED TO HANDLE NaN VALUES)

import pandas as pd
import json
import os
import joblib
import numpy as np # Import numpy to handle np.nan

def export_data_for_web():
    """
    Consolidates all necessary data into a single data.json file,
    correctly handling missing backtest values by converting them to null.
    """
    print("Starting data export for web...")

    try:
        predictions_df = pd.read_csv("latest_predictions.csv")
        summary_df = pd.read_csv("backtest_summary.csv")
    except FileNotFoundError as e:
        print(f"Error: Missing a required CSV file. {e}")
        print("Please run '3_automated_backtest.py' and '4_generate_predictions.py' first.")
        return

    # --- THE CRITICAL FIX IS HERE ---
    # Replace any NaN (Not a Number) values from the backtest summary with None.
    # json.dump will convert Python's None to the valid JSON value 'null'.
    # This prevents the literal string 'NaN' from being written to the file, which is invalid JSON.
    summary_df = summary_df.replace({np.nan: None})

    predictions_dict = predictions_df.to_dict(orient='records')
    summary_dict = summary_df.set_index('Ticker').to_dict(orient='index')

    print("Loading feature importances from models...")
    model_files = os.listdir('models')
    for ticker_model_file in model_files:
        if ticker_model_file.endswith('.pkl'):
            ticker = ticker_model_file.replace('_model.pkl', '')
            # Check if the ticker exists in the summary dictionary before proceeding
            if ticker in summary_dict:
                try:
                    payload = joblib.load(f"models/{ticker_model_file}")
                    # Add the importances to this ticker's summary data
                    if 'feature_importances' in payload and summary_dict[ticker] is not None:
                         summary_dict[ticker]['feature_importances'] = payload.get('feature_importances', {})
                except Exception as e:
                    print(f"Could not load feature importances for {ticker}: {e}")

    # Process price history
    price_history = {}
    stock_files = os.listdir('stock_data')
    for ticker_file in stock_files:
        if ticker_file.endswith('.csv'):
            ticker = ticker_file.split('.')[0]
            df = pd.read_csv(f"stock_data/{ticker_file}")
            df['Date'] = pd.to_datetime(df['Date'])
            df = df[df['Date'] > (pd.Timestamp.now() - pd.DateOffset(years=1))]
            df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
            price_history[ticker] = df.to_dict(orient='records')

    final_data = {
        "predictions": predictions_dict,
        "summary": summary_dict,
        "priceHistory": price_history
    }

    with open('data.json', 'w') as f:
        json.dump(final_data, f, indent=2)

    print("\nSuccessfully exported all data (including feature importances) to 'data.json'.")
    print("This file is now ready to be deployed with your static website.")

if __name__ == "__main__":
    export_data_for_web()