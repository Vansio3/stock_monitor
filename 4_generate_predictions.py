# 4_generate_predictions.py (UPDATED FOR TIER 2 FEATURES AND FIX)

import pandas as pd
import pandas_ta as ta
import numpy as np
import joblib
import os

def generate_all_predictions():
    """
    Loads all models, calculates the latest signal for each stock using
    advanced features, and saves all predictions to a single CSV file.
    """
    try:
        model_files = os.listdir('models')
        tickers = sorted([f.replace('_model.pkl', '') for f in model_files if f.endswith('.pkl')])
    except FileNotFoundError:
        print("Error: 'models' directory not found. Please train models first.")
        return

    try:
        spy_df = pd.read_csv("stock_data/SPY.csv", index_col="Date", parse_dates=True)
        spy_df.rename(columns={'Close': 'SPY_Close'}, inplace=True)
        spy_df['SPY_pct_change'] = spy_df['SPY_Close'].pct_change()
        spy_df.ta.rsi(close='SPY_Close', append=True, col_names=('SPY_RSI_14',))
        spy_df['SPY_RSI_change_1d'] = spy_df['SPY_RSI_14'].diff()
        market_features = spy_df[['SPY_pct_change', 'SPY_RSI_14', 'SPY_RSI_change_1d']].dropna()
        print("Successfully loaded market data (SPY) for predictions.")
    except FileNotFoundError:
        print("Warning: SPY.csv not found. Market context features will be skipped.")
        market_features = None

    all_predictions = []
    
    print(f"Generating predictions for {len(tickers)} stocks...")

    for ticker in tickers:
        try:
            model_payload = joblib.load(f"models/{ticker}_model.pkl")
            model = model_payload['model']
            
            # --- THE KEY FIX ---
            # Load the guaranteed correct feature order
            feature_order = model_payload['feature_order']

            df = pd.read_csv(f"stock_data/{ticker}.csv", index_col="Date", parse_dates=True)

            # --- CALCULATE ALL THE SAME FEATURES AS IN TRAINING ---
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
            df['volatility'] = df['Close'].pct_change().rolling(window=5).std() * np.sqrt(5)
            df.ta.cmf(high=df['High'], low=df['Low'], close=df['Close'], volume=df['Volume'], append=True)
            df['above_200_sma'] = (df['Close'] > df.ta.sma(200)).astype(int)

            if market_features is not None:
                df = df.join(market_features)
            
            # --- THE KEY FIX ---
            # Use the correct feature order to select data for prediction
            latest_data = df.iloc[-1:][feature_order]

            if latest_data.isnull().values.any():
                print(f"  Warning: Not enough recent data to calculate all features for {ticker}. Skipping.")
                continue

            prediction = model.predict(latest_data)[0]
            confidence_scores = model.predict_proba(latest_data)[0]
            confidence = confidence_scores[prediction]

            all_predictions.append({
                'Ticker': ticker,
                'Signal': prediction,
                'Confidence': confidence
            })
            
            signal_map = {0: 'SELL', 1: 'HOLD', 2: 'BUY'}
            print(f"  {ticker}: {signal_map.get(prediction, 'UNKNOWN')} ({confidence:.2%})")

        except Exception as e:
            print(f"Could not generate prediction for {ticker}: {e}")

    predictions_df = pd.DataFrame(all_predictions)
    predictions_df.to_csv("latest_predictions.csv", index=False)
    print("\nSuccessfully saved all predictions to latest_predictions.csv")

if __name__ == "__main__":
    generate_all_predictions()