# 4_generate_predictions.py (UPDATED FOR MORE FEATURES)

import pandas as pd
import pandas_ta as ta
import joblib
import os

def generate_all_predictions():
    """
    Loads all models, calculates the latest signal for each stock using
    enhanced features, and saves all predictions to a single CSV file.
    """
    try:
        model_files = os.listdir('models')
        tickers = sorted([f.replace('_model.pkl', '') for f in model_files if f.endswith('.pkl')])
    except FileNotFoundError:
        print("Error: 'models' directory not found. Please train models first.")
        return

    if not tickers:
        print("No models found.")
        return

    all_predictions = []
    # --- THIS LIST MUST EXACTLY MATCH THE ONE IN 2_model_trainer.py ---
    features_list = [
        'RSI_14', 'MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9',
        'BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0', 'ATRr_14',
        'STOCHk_14_3_3', 'STOCHd_14_3_3', 'OBV', 'ADX_14', 'WILLR_14',
        'RSI_change_1d', 'MACD_change_1d'
    ]
    
    print(f"Generating predictions for {len(tickers)} stocks...")

    for ticker in tickers:
        try:
            model = joblib.load(f"models/{ticker}_model.pkl")
            df = pd.read_csv(f"stock_data/{ticker}.csv", index_col="Date", parse_dates=True)

            # --- CALCULATE ALL THE SAME FEATURES ---
            df.ta.rsi(append=True)
            df.ta.macd(append=True)
            df.ta.bbands(length=20, append=True)
            df.ta.atr(append=True)
            df.ta.stoch(append=True)
            df.ta.obv(append=True)
            df.ta.adx(append=True)
            df.ta.willr(append=True)
            df['RSI_change_1d'] = df['RSI_14'].diff()
            df['MACD_change_1d'] = df['MACD_12_26_9'].diff()
            
            df.dropna(inplace=True)

            if df.empty:
                print(f"Not enough data for {ticker}, skipping.")
                continue

            latest_data = df.iloc[-1:][features_list]
            prediction = model.predict(latest_data)[0]
            confidence = model.predict_proba(latest_data)[0][prediction]

            all_predictions.append({
                'Ticker': ticker,
                'Signal': prediction,
                'Confidence': confidence
            })
            print(f"  {ticker}: {'BUY' if prediction == 1 else 'HOLD/SELL'} ({confidence:.2%})")

        except Exception as e:
            print(f"Could not generate prediction for {ticker}: {e}")

    # Save to CSV
    predictions_df = pd.DataFrame(all_predictions)
    predictions_df.to_csv("latest_predictions.csv", index=False)
    print("\nSuccessfully saved all predictions to latest_predictions.csv")

if __name__ == "__main__":
    generate_all_predictions()