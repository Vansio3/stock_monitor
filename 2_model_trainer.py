# 2_model_trainer.py (UPGRADED WITH TIER 1 ACCURACY & ROBUST LABELING)

import pandas as pd
import pandas_ta as ta
import numpy as np
import joblib
import os
import warnings

# Import new libraries for advanced modeling
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.exceptions import ConvergenceWarning

# Suppress warnings for a cleaner output
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning, module='lightgbm')


# --- Settings ---
try:
    TICKERS = sorted([f.split('.')[0] for f in os.listdir('stock_data') if f.endswith('.csv')])
except FileNotFoundError:
    print("Error: 'stock_data' directory not found. Please run '1_data_collector.py' first.")
    TICKERS = []

TARGET_DAYS_AHEAD = 5
MODELS_DIR = "models"
TEST_SET_PERCENTAGE = 0.2
# --- NEW: Quantile settings for labeling ---
BUY_QUANTILE = 0.80  # Top 20% of future returns will be labeled "Buy"
SELL_QUANTILE = 0.20 # Bottom 20% of future returns will be labeled "Sell"


# --- Code ---
def train_all_models():
    """
    Loops through all tickers, engineers features, uses robust quantile-based labeling,
    finds best hyperparameters with Time-Series CV, trains a LightGBM model,
    evaluates it, and saves the final model and its metadata.
    """
    if not TICKERS:
        print("No stock data found to train on. Exiting.")
        return

    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
        print(f"Created directory: {MODELS_DIR}")

    # --- Load and Prepare Market Data (SPY) ---
    try:
        spy_df = pd.read_csv("stock_data/SPY.csv", index_col="Date", parse_dates=True)
        spy_df.rename(columns={'Close': 'SPY_Close'}, inplace=True)
        spy_df['SPY_pct_change'] = spy_df['SPY_Close'].pct_change()
        spy_df.ta.rsi(close='SPY_Close', append=True, col_names=('SPY_RSI_14',))
        spy_df['SPY_RSI_change_1d'] = spy_df['SPY_RSI_14'].diff()
        market_features = spy_df[['SPY_pct_change', 'SPY_RSI_14', 'SPY_RSI_change_1d']].dropna()
        print("Successfully loaded and processed market data (SPY).")
    except FileNotFoundError:
        print("Warning: SPY.csv not found. Market context features will be skipped.")
        market_features = None

    print(f"Found data for {len(TICKERS)} stocks. Starting advanced training...")

    features_list = [
        'RSI_14', 'MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9',
        'BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0', 'ATRr_14',
        'STOCHk_14_3_3', 'STOCHd_14_3_3', 'OBV', 'ADX_14', 'WILLR_14',
        'RSI_change_1d', 'MACD_change_1d',
        'SPY_pct_change', 'SPY_RSI_14', 'SPY_RSI_change_1d',
        'volatility', 'CMF_20', 'above_200_sma'
    ]

    for ticker in TICKERS:
        if ticker == 'SPY':
            continue

        print(f"--- Processing {ticker} ---")

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
        df.ta.stoch(append=True)
        df.ta.obv(append=True)
        df.ta.adx(append=True)
        df.ta.willr(append=True)
        df['RSI_change_1d'] = df['RSI_14'].diff()
        df['MACD_change_1d'] = df['MACD_12_26_9'].diff()
        df['volatility'] = df['Close'].pct_change().rolling(window=TARGET_DAYS_AHEAD).std() * np.sqrt(TARGET_DAYS_AHEAD)
        df.ta.cmf(high=df['High'], low=df['Low'], close=df['Close'], volume=df['Volume'], append=True)
        df['above_200_sma'] = (df['Close'] > df.ta.sma(200)).astype(int)

        if market_features is not None:
            df = df.join(market_features)

        # --- ROBUST QUANTILE-BASED TARGET LABELING ---
        # Calculate the future return over the TARGET_DAYS_AHEAD period.
        df['future_return'] = df['Close'].pct_change(TARGET_DAYS_AHEAD).shift(-TARGET_DAYS_AHEAD)

        # Calculate the quantile thresholds on the entire dataset's future returns
        # This makes the labels adaptive to each stock's unique volatility profile.
        buy_threshold = df['future_return'].quantile(BUY_QUANTILE)
        sell_threshold = df['future_return'].quantile(SELL_QUANTILE)
        
        print(f"Labeling thresholds for {ticker}: Sell < {sell_threshold:.4f}, Buy > {buy_threshold:.4f}")

        conditions = [
            df['future_return'] > buy_threshold,  # If future return is in the top 20%, it's a "Buy"
            df['future_return'] < sell_threshold, # If future return is in the bottom 20%, it's a "Sell"
        ]
        choices = [2, 0] # 2 for Buy, 0 for Sell
        df['target'] = np.select(conditions, choices, default=1) # 1 for Hold (the middle 60%)
        
        # Check the distribution of the generated labels to confirm it worked
        print(f"Label distribution for {ticker}:\n{df['target'].value_counts(normalize=True)}")

        df.dropna(inplace=True)

        final_features = [f for f in features_list if f in df.columns]
        X = df[final_features]
        y = df['target']

        if len(X) < 100:
            print(f"Not enough data for {ticker} after feature generation, skipping.")
            continue

        split_index = int(len(X) * (1 - TEST_SET_PERCENTAGE))
        X_train, X_test = X[:split_index], X[split_index:]
        y_train, y_test = y[:split_index], y[split_index:]

        # --- MODEL TRAINING & HYPERPARAMETER TUNING ---
        print(f"Starting hyperparameter search for {ticker} with LightGBM...")
        # Use class_weight='balanced' to handle the natural imbalance in labels
        lgbm = lgb.LGBMClassifier(random_state=42, verbose=-1, class_weight='balanced')
        
        param_grid = {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.05, 0.1],
            'num_leaves': [20, 31, 40],
        }

        # Use TimeSeriesSplit for robust, chronologically-aware cross-validation
        tscv = TimeSeriesSplit(n_splits=5)

        grid_search = GridSearchCV(estimator=lgbm, param_grid=param_grid, cv=tscv, scoring='f1_weighted', n_jobs=-1, verbose=1)
        grid_search.fit(X_train, y_train)

        model = grid_search.best_estimator_
        print(f"Best parameters found: {grid_search.best_params_}")

        # --- EVALUATION ---
        predictions = model.predict(X_test)
        print(f"\n--- Evaluation Report for {ticker} (Test Set) ---")
        target_names = ['Sell (0)', 'Hold (1)', 'Buy (2)']
        
        print(classification_report(y_test, predictions, target_names=target_names, labels=[0, 1, 2], zero_division=0))

        # --- SAVE MODEL AND METADATA ---
        importances = pd.Series(model.feature_importances_, index=final_features)
        importances = importances.sort_values(ascending=False)
        
        model_payload = {
            'model': model,
            'feature_order': final_features,
            'feature_importances': importances.to_dict()
        }
        model_path = os.path.join(MODELS_DIR, f"{ticker}_model.pkl")
        joblib.dump(model_payload, model_path, compress=3)
        print(f"Model, feature order, and importances for {ticker} saved to {model_path}\n")

if __name__ == "__main__":
    train_all_models()