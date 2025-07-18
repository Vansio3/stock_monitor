# 2_model_trainer.py (UPDATED WITH TIER 2 IMPROVEMENTS & FIXES)

import pandas as pd
import pandas_ta as ta
import numpy as np
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
VOLATILITY_MULTIPLIER = 1.5 
MODELS_DIR = "models"
TEST_SET_PERCENTAGE = 0.2

# --- Code ---
def train_all_models():
    """
    Loops through all tickers, engineers advanced features, finds the best hyperparameters,
    trains a model, evaluates it, saves the model AND its feature order/importances.
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
        'volatility', 'CMF_20', 'above_200_sma' # CMF_20 is the default name from pandas-ta
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

        # --- ADVANCED FEATURE ENGINEERING ---
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

        df['future_price'] = df['Close'].shift(-TARGET_DAYS_AHEAD)
        df['price_change'] = (df['future_price'] - df['Close'])
        dynamic_threshold = df['ATRr_14'] / 100 * df['Close'] * VOLATILITY_MULTIPLIER

        conditions = [
            df['price_change'] > dynamic_threshold,
            df['price_change'] < -dynamic_threshold,
        ]
        choices = [2, 0]
        df['target'] = np.select(conditions, choices, default=1)
        
        df.dropna(inplace=True)
        
        final_features = [f for f in features_list if f in df.columns]
        missing_features = set(features_list) - set(final_features)
        if missing_features:
            print(f"Warning: Missing features for {ticker}, they will be excluded: {missing_features}")

        X = df[final_features]
        y = df['target']

        if len(X) < 50 or y.nunique() < 2:
            print(f"Not enough data or outcome variability for {ticker}, skipping.")
            continue

        split_index = int(len(X) * (1 - TEST_SET_PERCENTAGE))
        X_train, X_test = X[:split_index], X[split_index:]
        y_train, y_test = y[:split_index], y[split_index:]

        print(f"Starting hyperparameter search for {ticker}...")
        param_grid = {
            'n_estimators': [100, 200], 'max_depth': [10, 20, None],
            'min_samples_leaf': [2, 4], 'max_features': ['sqrt', 'log2'],
        }
        rf = RandomForestClassifier(random_state=42)
        
        cv_splits = 3
        if y_train.value_counts().min() < cv_splits:
             cv_splits = y_train.value_counts().min()

        if cv_splits < 2:
            print(f"Cannot perform cross-validation for {ticker} due to extreme class imbalance. Skipping.")
            continue

        grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=cv_splits, scoring='f1_weighted', n_jobs=-1, verbose=0)
        grid_search.fit(X_train, y_train)

        model = grid_search.best_estimator_
        print(f"Best parameters found: {grid_search.best_params_}")

        predictions = model.predict(X_test)
        print(f"\n--- Evaluation Report for {ticker} (Test Set) ---")
        target_names = ['Sell (0)', 'Hold (1)', 'Buy (2)']
        
        print(classification_report(y_test, predictions, target_names=target_names, labels=[0, 1, 2], zero_division=0))

        importances = pd.Series(model.feature_importances_, index=final_features)
        importances = importances.sort_values(ascending=False)
        
        # --- THE KEY FIX ---
        # Save the model, the exact feature order, and the sorted importances
        model_payload = {
            'model': model,
            'feature_order': final_features,  # THIS IS THE CRITICAL FIX
            'feature_importances': importances.to_dict()
        }
        model_path = os.path.join(MODELS_DIR, f"{ticker}_model.pkl")
        joblib.dump(model_payload, model_path, compress=3)
        print(f"Model, feature order, and importances for {ticker} saved to {model_path}\n")

if __name__ == "__main__":
    train_all_models()