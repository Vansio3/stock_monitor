# 2_model_trainer.py (UPDATED WITH FASTER RANDOMIZEDSEARCHCV)

import pandas as pd
import pandas_ta as ta
import numpy as np
import joblib
import os
import warnings

# Import new libraries for advanced modeling
import lightgbm as lgb
# --- *** UPDATED IMPORT *** ---
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV # Changed GridSearchCV to RandomizedSearchCV
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
TEST_SET_PERCENTAGE = 0.3
BUY_QUANTILE = 0.80
SELL_QUANTILE = 0.20


# --- Code ---
def train_all_models():
    """
    Loops through all tickers, engineers features, uses robust quantile-based labeling,
    finds best hyperparameters with Time-Series CV using RandomizedSearchCV,
    trains a LightGBM model, evaluates it, and saves the final model.
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
        'volatility', 'CMF_20', 'above_200_sma',
        'sma_trend_strength', 'distance_from_sma_200', 'RSI_volatility', 'ROC_21'
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
        df.ta.cmf(high=df['High'], low=df['Low'], close=df['Close'], volume=df['Volume'], append=True)
        df.ta.roc(length=21, append=True)
        df['RSI_change_1d'] = df['RSI_14'].diff()
        df['MACD_change_1d'] = df['MACD_12_26_9'].diff()
        df['volatility'] = df['Close'].pct_change().rolling(window=TARGET_DAYS_AHEAD).std() * np.sqrt(TARGET_DAYS_AHEAD)
        
        sma50 = df.ta.sma(50)
        sma200 = df.ta.sma(200)
        df['above_200_sma'] = (df['Close'] > sma200).astype(int)
        df['sma_trend_strength'] = (sma50 > sma200).astype(int)
        df['distance_from_sma_200'] = (df['Close'] - sma200) / sma200
        df['RSI_volatility'] = df['RSI_14'].rolling(window=20).std()

        if market_features is not None:
            df = df.join(market_features)

        # --- ROBUST, LEAK-FREE LABELING ---
        df['future_return'] = df['Close'].pct_change(TARGET_DAYS_AHEAD).shift(-TARGET_DAYS_AHEAD)
        df.dropna(inplace=True)
        
        split_index = int(len(df) * (1 - TEST_SET_PERCENTAGE))
        
        train_returns = df['future_return'].iloc[:split_index]
        buy_threshold = train_returns.quantile(BUY_QUANTILE)
        sell_threshold = train_returns.quantile(SELL_QUANTILE)
        
        print(f"Labeling thresholds for {ticker} (from training data): Sell < {sell_threshold:.4f}, Buy > {buy_threshold:.4f}")

        conditions = [
            df['future_return'] > buy_threshold,
            df['future_return'] < sell_threshold,
        ]
        choices = [2, 0]
        df['target'] = np.select(conditions, choices, default=1)
        
        print(f"Label distribution for {ticker}:\n{df['target'].value_counts(normalize=True)}")

        # --- PREPARE DATA FOR MODELING ---
        final_features = [f for f in features_list if f in df.columns]
        X = df[final_features]
        y = df['target']

        if len(X) < 100:
            print(f"Not enough data for {ticker} after processing, skipping.")
            continue
        
        X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
        y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

        # --- MODEL TRAINING & HYPERPARAMETER TUNING ---
        print(f"Starting hyperparameter search for {ticker} with LightGBM...")
        lgbm = lgb.LGBMClassifier(random_state=42, verbose=-1, class_weight='balanced')
        
        param_grid = {
            'n_estimators': [100, 200, 300, 400],
            'learning_rate': [0.01, 0.05, 0.1],
            'num_leaves': [20, 31, 40, 50],
            'reg_alpha': [0.0, 0.1, 0.5],
            'reg_lambda': [0.0, 0.1, 0.5]
        }

        tscv = TimeSeriesSplit(n_splits=3) # Reduced folds for speed
        
        # --- *** UPDATED: Using RandomizedSearchCV for efficiency *** ---
        # It will test 25 random combinations instead of all 432 possibilities.
        random_search = RandomizedSearchCV(
            estimator=lgbm, 
            param_distributions=param_grid, 
            n_iter=25,  # Number of parameter settings that are sampled
            cv=tscv, 
            scoring='f1_weighted', 
            n_jobs=-1, 
            verbose=1, 
            random_state=42
        )
        random_search.fit(X_train, y_train)

        model = random_search.best_estimator_
        print(f"Best parameters found: {random_search.best_params_}")

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