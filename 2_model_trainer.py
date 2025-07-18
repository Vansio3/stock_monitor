# 2_model_trainer.py (UPDATED WITH TIER 1 IMPROVEMENTS)

import pandas as pd
import pandas_ta as ta
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report # Replaces accuracy_score
from sklearn.model_selection import GridSearchCV # New import for tuning
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

    print(f"Found data for {len(TICKERS)} stocks. Starting hyperparameter tuning and training...")

    features_list = [
        'RSI_14', 'MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9',
        'BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0', 'ATRr_14',
        'STOCHk_14_3_3', 'STOCHd_14_3_3', 'OBV', 'ADX_14', 'WILLR_14',
        'RSI_change_1d', 'MACD_change_1d'
    ]

    for ticker in TICKERS:
        print(f"--- Processing {ticker} ---")

        try:
            df = pd.read_csv(f"stock_data/{ticker}.csv", index_col="Date", parse_dates=True)
        except FileNotFoundError:
            print(f"Data for {ticker} not found, skipping.")
            continue

        # --- FEATURE ENGINEERING (Same as before) ---
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
        df['future_price'] = df['Close'].shift(-TARGET_DAYS_AHEAD)
        df['price_change'] = (df['future_price'] - df['Close']) / df['Close']
        df['target'] = (df['price_change'] > PREDICTION_THRESHOLD).astype(int)
        df.dropna(inplace=True)

        X = df[features_list]
        y = df['target']

        if len(X) < 50:
            print(f"Not enough data for {ticker} after processing, skipping.")
            continue

        split_index = int(len(X) * (1 - TEST_SET_PERCENTAGE))
        X_train, X_test = X[:split_index], X[split_index:]
        y_train, y_test = y[:split_index], y[split_index:]

        # --- HYPERPARAMETER TUNING (NEW) ---
        print(f"Starting hyperparameter search for {ticker}...")
        
        # 1. Define the grid of parameters to test
        param_grid = {
            'n_estimators': [100, 200],         # Number of trees in the forest
            'max_depth': [10, 20, None],       # Maximum depth of the trees
            'min_samples_leaf': [2, 4],          # Minimum samples required at a leaf node
            'max_features': ['sqrt', 'log2'],  # Number of features to consider for best split
        }

        # 2. Set up the GridSearchCV object
        rf = RandomForestClassifier(random_state=42)
        grid_search = GridSearchCV(
            estimator=rf,
            param_grid=param_grid,
            cv=3,                      # 3-fold cross-validation
            scoring='f1',              # <<< Evaluate based on F1-score, not accuracy
            n_jobs=-1,                 # Use all available CPU cores
            verbose=1                  # Show progress
        )

        # 3. Run the search on the training data
        grid_search.fit(X_train, y_train)

        # 4. The best model is found by the search
        model = grid_search.best_estimator_
        print(f"Best parameters found: {grid_search.best_params_}")

        # --- EVALUATE MODEL (IMPROVED) ---
        predictions = model.predict(X_test)
        
        # Use classification_report for a detailed breakdown
        print(f"\n--- Evaluation Report for {ticker} (Test Set) ---")
        # target_names are mapped to the labels 0 and 1
        print(classification_report(y_test, predictions, target_names=['Hold/Sell (0)', 'Buy (1)'], zero_division=0))

        # --- Save the best trained model ---
        model_path = os.path.join(MODELS_DIR, f"{ticker}_model.pkl")
        joblib.dump(model, model_path, compress=3)
        print(f"Model for {ticker} saved to {model_path}\n")

if __name__ == "__main__":
    train_all_models()