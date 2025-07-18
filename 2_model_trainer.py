# 2_model_trainer.py (UPDATED WITH TIER 1 IMPROVEMENTS)

import pandas as pd
import pandas_ta as ta
import numpy as np ### TIER 1: Import numpy for the new target logic
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
import joblib
import os
import warnings

# Suppress ConvergenceWarning from scikit-learn, which can be noisy during grid search
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

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
    """
    Loops through all discovered tickers, finds the best hyperparameters using GridSearchCV,
    trains a model on those parameters, evaluates it with a detailed report, and saves it.
    """
    if not TICKERS:
        print("No stock data found to train on. Exiting.")
        return

    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
        print(f"Created directory: {MODELS_DIR}")

    ### TIER 1: LOAD AND PREPARE MARKET DATA (SPY) ###
    try:
        spy_df = pd.read_csv("stock_data/SPY.csv", index_col="Date", parse_dates=True)
        # Use a unique name for SPY's close to avoid conflicts
        spy_df.rename(columns={'Close': 'SPY_Close'}, inplace=True)
        # Calculate market features
        spy_df['SPY_pct_change'] = spy_df['SPY_Close'].pct_change()
        spy_df.ta.rsi(close='SPY_Close', append=True, col_names=('SPY_RSI_14',))
        # Keep only the features we need, and drop any initial NaNs
        market_features = spy_df[['SPY_pct_change', 'SPY_RSI_14']].dropna()
        print("Successfully loaded and processed market data (SPY).")
    except FileNotFoundError:
        print("Warning: SPY.csv not found. Market context features will be skipped.")
        market_features = None
    
    print(f"Found data for {len(TICKERS)} stocks. Starting hyperparameter tuning and training...")

    features_list = [
        'RSI_14', 'MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9',
        'BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0', 'ATRr_14',
        'STOCHk_14_3_3', 'STOCHd_14_3_3', 'OBV', 'ADX_14', 'WILLR_14',
        'RSI_change_1d', 'MACD_change_1d'
    ]
    ### TIER 1: ADD NEW MARKET FEATURES TO THE LIST ###
    if market_features is not None:
        features_list.extend(['SPY_pct_change', 'SPY_RSI_14'])

    for ticker in TICKERS:
        ### TIER 1: Skip training a model for the benchmark itself ###
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
        
        ### TIER 1: JOIN WITH MARKET DATA ###
        if market_features is not None:
            df = df.join(market_features)

        df['future_price'] = df['Close'].shift(-TARGET_DAYS_AHEAD)
        df['price_change'] = (df['future_price'] - df['Close']) / df['Close']

        ### TIER 1: REDEFINE TARGET FOR 3-CLASS CLASSIFICATION (SELL/HOLD/BUY) ###
        conditions = [
            df['price_change'] > PREDICTION_THRESHOLD,   # Condition for Buy
            df['price_change'] < -PREDICTION_THRESHOLD,  # Condition for Sell
        ]
        # Outcomes: 2 for Buy, 0 for Sell. Default is 1 (Hold).
        choices = [2, 0] 
        df['target'] = np.select(conditions, choices, default=1)
        
        df.dropna(inplace=True)

        X = df[features_list]
        y = df['target']

        if len(X) < 50:
            print(f"Not enough data for {ticker} after processing, skipping.")
            continue

        split_index = int(len(X) * (1 - TEST_SET_PERCENTAGE))
        X_train, X_test = X[:split_index], X[split_index:]
        y_train, y_test = y[:split_index], y[split_index:]

        # --- HYPERPARAMETER TUNING ---
        print(f"Starting hyperparameter search for {ticker}...")
        
        param_grid = {
            'n_estimators': [100, 200],         
            'max_depth': [10, 20, None],       
            'min_samples_leaf': [2, 4],          
            'max_features': ['sqrt', 'log2'],
        }

        rf = RandomForestClassifier(random_state=42)
        grid_search = GridSearchCV(
            estimator=rf,
            param_grid=param_grid,
            cv=3,
            ### TIER 1: Use a metric suitable for multi-class, imbalanced data ###
            scoring='f1_weighted',
            n_jobs=-1,
            verbose=1
        )

        grid_search.fit(X_train, y_train)

        model = grid_search.best_estimator_
        print(f"Best parameters found: {grid_search.best_params_}")

        # --- EVALUATE MODEL ---
        predictions = model.predict(X_test)
        
        print(f"\n--- Evaluation Report for {ticker} (Test Set) ---")
        ### TIER 1: Update target names for the 3 classes ###
        target_names = ['Sell (0)', 'Hold (1)', 'Buy (2)']
        print(classification_report(y_test, predictions, target_names=target_names, zero_division=0))

        # --- Save the best trained model ---
        model_path = os.path.join(MODELS_DIR, f"{ticker}_model.pkl")
        joblib.dump(model, model_path, compress=3)
        print(f"Model for {ticker} saved to {model_path}\n")

if __name__ == "__main__":
    train_all_models()