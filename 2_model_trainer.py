# 2_model_trainer.py

import pandas as pd
import pandas_ta as ta
# We no longer need train_test_split from sklearn
# from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

# --- Settings ---
# Automatically discover tickers by looking at the files in the stock_data folder.
try:
    TICKERS = sorted([f.split('.')[0] for f in os.listdir('stock_data') if f.endswith('.csv')])
except FileNotFoundError:
    print("Error: 'stock_data' directory not found. Please run '1_data_collector.py' first.")
    TICKERS = [] 

TARGET_DAYS_AHEAD = 5
PREDICTION_THRESHOLD = 0.02
MODELS_DIR = "models"
TEST_SET_PERCENTAGE = 0.2 # Use 20% of the data for testing

# --- Code ---
def train_all_models():
    """Loops through all discovered tickers, trains a model, and saves it."""
    if not TICKERS:
        print("No stock data found to train on. Exiting.")
        return

    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
        print(f"Created directory: {MODELS_DIR}")

    print(f"Found data for {len(TICKERS)} stocks. Starting training process...")

    for ticker in TICKERS:
        print(f"--- Training model for {ticker} ---")
        
        try:
            df = pd.read_csv(f"stock_data/{ticker}.csv", index_col="Date", parse_dates=True)
        except FileNotFoundError:
            print(f"Data for {ticker} not found, skipping.")
            continue

        # --- FEATURE ENGINEERING ---
        df.ta.rsi(append=True)
        df.ta.macd(append=True)
        df.ta.bbands(length=20, append=True)
        df.ta.atr(append=True)

        # Define the Target Variable
        df['future_price'] = df['Close'].shift(-TARGET_DAYS_AHEAD)
        df['price_change'] = (df['future_price'] - df['Close']) / df['Close']
        df['target'] = (df['price_change'] > PREDICTION_THRESHOLD).astype(int)

        # Prepare Data for Model
        df.dropna(inplace=True)
        
        feature_prefixes = ['RSI', 'MACD', 'BBL', 'BBM', 'BBU', 'ATRr']
        features = [col for col in df.columns if any(col.startswith(prefix) for prefix in feature_prefixes)]
        
        X = df[features]
        y = df['target']
        
        if len(X) < 50:
            print(f"Not enough data for {ticker} after processing, skipping.")
            continue

        # --- THE FIX: Chronological Time-Series Split ---
        # Instead of a random split, we split the data by date.
        # The first (1 - TEST_SET_PERCENTAGE) of data is for training.
        # The last TEST_SET_PERCENTAGE of data is for testing.
        split_index = int(len(X) * (1 - TEST_SET_PERCENTAGE))
        
        X_train, X_test = X[:split_index], X[split_index:]
        y_train, y_test = y[:split_index], y[split_index:]
        # --- End of Fix ---
        
        # We removed the old line:
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
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