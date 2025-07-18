# 2_model_trainer.py (UPDATED FOR MORE FEATURES)

import pandas as pd
import pandas_ta as ta
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

# --- Settings ---
try:
    TICKERS = sorted([f.split('.')[0] for f in os.listdir('stock_data') if f.endswith('.csv')])
except FileNotFoundError:
    print("Error: 'stock_data' directory not found. Please run '1_data_collector.py' first.")
    TICKERS = [] 

TARGET_DAYS_AHEAD = 5
PREDICTION_THRESHOLD = 0.02
MODELS_DIR = "models"
TEST_SET_PERCENTAGE = 0.2

# --- Code ---
def train_all_models():
    """Loops through all discovered tickers, trains a model with enhanced features, and saves it."""
    if not TICKERS:
        print("No stock data found to train on. Exiting.")
        return

    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
        print(f"Created directory: {MODELS_DIR}")

    print(f"Found data for {len(TICKERS)} stocks. Starting training process with enhanced features...")

    # --- THIS IS THE NEW, EXPANDED LIST OF FEATURES THE MODEL WILL USE ---
    # We define it once here to ensure consistency.
    features_list = [
        'RSI_14', 'MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9',
        'BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0', 'ATRr_14',
        'STOCHk_14_3_3', 'STOCHd_14_3_3', 'OBV', 'ADX_14', 'WILLR_14',
        'RSI_change_1d', 'MACD_change_1d'
    ]

    for ticker in TICKERS:
        print(f"--- Training model for {ticker} ---")
        
        try:
            df = pd.read_csv(f"stock_data/{ticker}.csv", index_col="Date", parse_dates=True)
        except FileNotFoundError:
            print(f"Data for {ticker} not found, skipping.")
            continue

        # --- FEATURE ENGINEERING (EXPANDED) ---
        # 1. Standard Indicators
        df.ta.rsi(append=True)
        df.ta.macd(append=True)
        df.ta.bbands(length=20, append=True)
        df.ta.atr(append=True)
        df.ta.stoch(append=True)
        df.ta.obv(append=True)
        df.ta.adx(append=True)
        df.ta.willr(append=True)
        
        # 2. Lag/Difference Features
        df['RSI_change_1d'] = df['RSI_14'].diff()
        df['MACD_change_1d'] = df['MACD_12_26_9'].diff()

        # Define the Target Variable
        df['future_price'] = df['Close'].shift(-TARGET_DAYS_AHEAD)
        df['price_change'] = (df['future_price'] - df['Close']) / df['Close']
        df['target'] = (df['price_change'] > PREDICTION_THRESHOLD).astype(int)

        # Prepare Data for Model
        df.dropna(inplace=True)
        
        # We now use our predefined features_list
        X = df[features_list]
        y = df['target']
        
        if len(X) < 50:
            print(f"Not enough data for {ticker} after processing, skipping.")
            continue

        # Chronological Time-Series Split
        split_index = int(len(X) * (1 - TEST_SET_PERCENTAGE))
        X_train, X_test = X[:split_index], X[split_index:]
        y_train, y_test = y[:split_index], y[split_index:]
        
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Evaluate Model
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        print(f"Accuracy for {ticker}: {accuracy:.2%}")

        # Save the trained model
        model_path = os.path.join(MODELS_DIR, f"{ticker}_model.pkl")
        joblib.dump(model, model_path, compress=3)
        print(f"Model for {ticker} saved to {model_path}\n")

if __name__ == "__main__":
    train_all_models()