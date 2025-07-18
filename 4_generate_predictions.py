# 4_generate_predictions.py (UPDATED FOR MORE FEATURES AND MARKET CONTEXT)

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

    ### TIER 1: LOAD AND PREPARE MARKET DATA (SPY) FOR PREDICTION ###
    try:
        spy_df = pd.read_csv("stock_data/SPY.csv", index_col="Date", parse_dates=True)
        spy_df.rename(columns={'Close': 'SPY_Close'}, inplace=True)
        spy_df['SPY_pct_change'] = spy_df['SPY_Close'].pct_change()
        spy_df.ta.rsi(close='SPY_Close', append=True, col_names=('SPY_RSI_14',))
        market_features = spy_df[['SPY_pct_change', 'SPY_RSI_14']].dropna()
        print("Successfully loaded market data (SPY) for predictions.")
    except FileNotFoundError:
        print("Warning: SPY.csv not found. Market context features will be skipped.")
        market_features = None

    all_predictions = []
    # --- THIS LIST MUST EXACTLY MATCH THE ONE IN 2_model_trainer.py ---
    features_list = [
        'RSI_14', 'MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9',
        'BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0', 'ATRr_14',
        'STOCHk_14_3_3', 'STOCHd_14_3_3', 'OBV', 'ADX_14', 'WILLR_14',
        'RSI_change_1d', 'MACD_change_1d',
        ### TIER 1: ADD NEW MARKET FEATURES TO THE LIST ###
        'SPY_pct_change', 'SPY_RSI_14'
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

            ### TIER 1: JOIN WITH MARKET DATA ###
            if market_features is not None:
                df = df.join(market_features)
            
            df.dropna(inplace=True)

            if df.empty:
                print(f"Not enough data for {ticker}, skipping.")
                continue

            latest_data = df.iloc[-1:][features_list]
            prediction = model.predict(latest_data)[0]
            confidence = model.predict_proba(latest_data)[0][prediction]

            all_predictions.append({
                'Ticker': ticker,
                'Signal': prediction, # Will be 0, 1, or 2
                'Confidence': confidence
            })
            
            ### TIER 1: MAP PREDICTION TO HUMAN-READABLE SIGNAL ###
            signal_map = {0: 'SELL', 1: 'HOLD', 2: 'BUY'}
            print(f"  {ticker}: {signal_map.get(prediction, 'UNKNOWN')} ({confidence:.2%})")

        except Exception as e:
            print(f"Could not generate prediction for {ticker}: {e}")

    # Save to CSV
    predictions_df = pd.DataFrame(all_predictions)
    predictions_df.to_csv("latest_predictions.csv", index=False)
    print("\nSuccessfully saved all predictions to latest_predictions.csv")

if __name__ == "__main__":
    generate_all_predictions()