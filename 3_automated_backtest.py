# 3_automated_backtest.py (UPDATED WITH MORE REALISTIC TRADING LOGIC & EXECUTION)

import pandas as pd
import pandas_ta as ta
from backtesting import Backtest, Strategy
import joblib
import os
import numpy as np
import traceback

# --- BACKTESTING SETTINGS ---
STOP_LOSS_PCT = 0.05

def indicator_func(func, *args, **kwargs):
    input_series = args[0]
    result = func(*args, **kwargs)
    if isinstance(result, pd.DataFrame):
        return tuple(res_col.reindex(input_series.index) for _, res_col in result.items())
    else:
        return result.reindex(input_series.index)

def identity_func(series):
    return series

def rolling_volatility(series, window, target_days):
    return series.pct_change().rolling(window=window).std() * np.sqrt(target_days)

def rsi_volatility_func(series: pd.Series, rsi_window=14, vol_window=20) -> pd.Series:
    """Calculates RSI and then the rolling standard deviation of that RSI."""
    rsi = ta.rsi(series, length=rsi_window)
    return rsi.rolling(window=vol_window).std()

def run_all_backtests():
    try:
        model_files = os.listdir('models')
        tickers = sorted([f.split('_model.pkl')[0] for f in model_files if f.endswith('.pkl')])
    except FileNotFoundError:
        print("Error: 'models' directory not found.")
        return

    try:
        spy_df = pd.read_csv("stock_data/SPY.csv", index_col="Date", parse_dates=True)
        spy_df.rename(columns={'Close': 'SPY_Close'}, inplace=True)
        spy_df['SPY_pct_change'] = spy_df['SPY_Close'].pct_change()
        spy_df.ta.rsi(close='SPY_Close', append=True, col_names=('SPY_RSI_14',))
        spy_df['SPY_RSI_change_1d'] = spy_df['SPY_RSI_14'].diff()
        market_features = spy_df[['SPY_pct_change', 'SPY_RSI_14', 'SPY_RSI_change_1d']]
        print("Backtester: Successfully loaded market data (SPY).")
    except FileNotFoundError:
        print("Backtester Warning: SPY.csv not found.")
        market_features = None

    all_stats = []
    print(f"Found models for: {', '.join(tickers)}. Running backtests...")

    for ticker in tickers:
        print(f"--- Backtesting for {ticker} ---")
        try:
            model_payload = joblib.load(f"models/{ticker}_model.pkl")
            model = model_payload['model']
            feature_order = model_payload['feature_order']

            class AiStrategy(Strategy):
                def init(self):
                    close_series = pd.Series(self.data.Close, index=self.data.index)
                    high_series = pd.Series(self.data.High, index=self.data.index)
                    low_series = pd.Series(self.data.Low, index=self.data.index)
                    volume_series = pd.Series(self.data.Volume, index=self.data.index)

                    self.rsi = self.I(indicator_func, ta.rsi, close_series)
                    self.willr = self.I(indicator_func, ta.willr, high_series, low_series, close_series)
                    self.obv = self.I(indicator_func, ta.obv, close_series, volume_series)
                    self.atr = self.I(indicator_func, ta.atr, high_series, low_series, close_series)
                    self.macd, self.macd_hist, self.macd_signal = self.I(indicator_func, ta.macd, close_series)
                    self.bbl, self.bbm, self.bbu, _, _ = self.I(indicator_func, ta.bbands, close_series, length=20)
                    self.stochk, self.stochd = self.I(indicator_func, ta.stoch, high_series, low_series, close_series)
                    self.adx, _, _ = self.I(indicator_func, ta.adx, high_series, low_series, close_series)
                    self.cmf = self.I(indicator_func, ta.cmf, high_series, low_series, close_series, volume_series)
                    self.volatility = self.I(rolling_volatility, close_series, window=5, target_days=5)
                    self.sma50 = self.I(indicator_func, ta.sma, close_series, length=50)
                    self.sma200 = self.I(indicator_func, ta.sma, close_series, length=200)
                    self.roc21 = self.I(indicator_func, ta.roc, close_series, length=21)
                    self.rsi_volatility = self.I(rsi_volatility_func, close_series)

                    if 'SPY_pct_change' in self.data.df.columns:
                        self.spy_pct_change = self.I(identity_func, self.data.SPY_pct_change)
                        self.spy_rsi = self.I(identity_func, self.data.SPY_RSI_14)
                        self.spy_rsi_change = self.I(identity_func, self.data.SPY_RSI_change_1d)
                    else:
                        nan_series = pd.Series([np.nan] * len(self.data.Close), index=self.data.index)
                        self.spy_pct_change, self.spy_rsi, self.spy_rsi_change = nan_series, nan_series, nan_series
                
                # --- *** UPDATED TRADING LOGIC *** ---
                def next(self):
                    # Do not trade if there is not enough data to calculate indicators
                    if len(self.data.Close) < 210: # A safe buffer for a 200-day SMA
                        return

                    feature_values = {
                        'RSI_14': self.rsi[-1], 'MACD_12_26_9': self.macd[-1], 'MACDh_12_26_9': self.macd_hist[-1],
                        'MACDs_12_26_9': self.macd_signal[-1], 'BBL_20_2.0': self.bbl[-1], 'BBM_20_2.0': self.bbm[-1],
                        'BBU_20_2.0': self.bbu[-1], 'ATRr_14': self.atr[-1], 'STOCHk_14_3_3': self.stochk[-1],
                        'STOCHd_14_3_3': self.stochd[-1], 'OBV': self.obv[-1], 'ADX_14': self.adx[-1], 'WILLR_14': self.willr[-1],
                        'RSI_change_1d': self.rsi[-1] - self.rsi[-2],
                        'MACD_change_1d': self.macd[-1] - self.macd[-2],
                        'SPY_pct_change': self.spy_pct_change[-1], 'SPY_RSI_14': self.spy_rsi[-1],
                        'SPY_RSI_change_1d': self.spy_rsi_change[-1],
                        'CMF_20': self.cmf[-1],
                        'above_200_sma': 1 if self.data.Close[-1] > self.sma200[-1] else 0,
                        'volatility': self.volatility[-1],
                        'sma_trend_strength': 1 if self.sma50[-1] > self.sma200[-1] else 0,
                        'distance_from_sma_200': (self.data.Close[-1] - self.sma200[-1]) / self.sma200[-1],
                        'RSI_volatility': self.rsi_volatility[-1],
                        'ROC_21': self.roc21[-1]
                    }
                    features = pd.DataFrame([feature_values])[feature_order]

                    # Ensure all features are present and not NaN
                    if features.isnull().values.any():
                        return

                    prediction = model.predict(features)[0]

                    # If we are not in a position, only enter on a BUY signal
                    if not self.position and prediction == 2:
                        self.buy(sl=self.data.Close[-1] * (1 - STOP_LOSS_PCT))
                    
                    # If we are in a position, exit if the signal is no longer BUY
                    elif self.position and prediction != 2:
                        self.position.close()
            
            df = pd.read_csv(f"stock_data/{ticker}.csv", index_col="Date", parse_dates=True)
            if market_features is not None:
                df = df.join(market_features)

            # --- *** MORE REALISTIC BACKTEST EXECUTION SETTINGS *** ---
            # trade_on_close=False ensures trades execute on the next bar's open,
            # preventing the model from seeing the close price of the bar it's trading on.
            bt = Backtest(df, AiStrategy, 
                          cash=10000, 
                          commission=.002,
                          trade_on_close=False,
                          exclusive_orders=True)
                          
            stats = bt.run()
            stats['Ticker'] = ticker
            all_stats.append(stats)
        except Exception as e:
            print(f"An error occurred while backtesting {ticker}:")
            traceback.print_exc()

    if not all_stats:
        print("No backtests were successfully completed.")
        return

    results_df = pd.DataFrame(all_stats)
    results_df.set_index('Ticker', inplace=True)
    columns_to_show = {
        'Return [%]': 'Return %', 'Sharpe Ratio': 'Sharpe Ratio', 'Max. Drawdown [%]': 'Max Drawdown %',
        'Win Rate [%]': 'Win Rate %', '# Trades': '# Trades', 'Buy & Hold Return [%]': 'Buy & Hold %'
    }
    summary_df = results_df[list(columns_to_show.keys())].rename(columns=columns_to_show)
    summary_df = summary_df.sort_values(by='Sharpe Ratio', ascending=False)
    print("\n\n--- Automated Backtest Summary (Advanced Features + Stop-Loss) ---")
    print(summary_df)
    summary_df.to_csv("backtest_summary.csv")
    print("\nSummary saved to backtest_summary.csv")

if __name__ == "__main__":
    run_all_backtests()