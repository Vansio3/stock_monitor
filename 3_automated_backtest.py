# 3_automated_backtest.py (UPDATED FOR TIER 1 IMPROVEMENTS)

import pandas as pd
import pandas_ta as ta
from backtesting import Backtest, Strategy
import joblib
import os

def indicator_func(func, *args, **kwargs):
    """
    A wrapper to ensure indicator outputs are Series with the same index
    as the input data, padded with NaNs.
    """
    input_series = args[0]
    result = func(*args, **kwargs)
    
    if isinstance(result, pd.DataFrame):
        return tuple(
            res_col.reindex(input_series.index) for _, res_col in result.items()
        )
    else:
        return result.reindex(input_series.index)

### TIER 1: Helper function to pass pre-calculated columns to the strategy ###
def identity_func(series):
    return series

def run_all_backtests():
    """Finds all models, runs a backtest for each, and prints a summary."""
    
    try:
        model_files = os.listdir('models')
        tickers = sorted([f.split('_model.pkl')[0] for f in model_files if f.endswith('.pkl')])
    except FileNotFoundError:
        print("Error: 'models' directory not found. Please train models first using '2_model_trainer.py'.")
        return

    if not tickers:
        print("No models found in the 'models' directory.")
        return

    ### TIER 1: LOAD MARKET DATA ONCE FOR ALL BACKTESTS ###
    try:
        spy_df = pd.read_csv("stock_data/SPY.csv", index_col="Date", parse_dates=True)
        spy_df.rename(columns={'Close': 'SPY_Close'}, inplace=True)
        spy_df['SPY_pct_change'] = spy_df['SPY_Close'].pct_change()
        spy_df.ta.rsi(close='SPY_Close', append=True, col_names=('SPY_RSI_14',))
        market_features = spy_df[['SPY_pct_change', 'SPY_RSI_14']]
        print("Backtester: Successfully loaded market data (SPY).")
    except FileNotFoundError:
        print("Backtester Warning: SPY.csv not found. Market context features will be skipped.")
        market_features = None

    all_stats = []
    print(f"Found models for: {', '.join(tickers)}. Running backtests...")

    for ticker in tickers:
        print(f"--- Backtesting for {ticker} ---")
        try:
            model = joblib.load(f"models/{ticker}_model.pkl")

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
                    self.macd_line, self.macd_hist, self.macd_signal = self.I(indicator_func, ta.macd, close_series)
                    self.bbl, self.bbm, self.bbu, _, _ = self.I(indicator_func, ta.bbands, close_series, length=20)
                    self.stochk, self.stochd = self.I(indicator_func, ta.stoch, high_series, low_series, close_series)
                    self.adx, _, _ = self.I(indicator_func, ta.adx, high_series, low_series, close_series)

                    ### TIER 1: Access market data passed into the strategy ###
                    if 'SPY_pct_change' in self.data.df.columns:
                        self.spy_pct_change = self.I(identity_func, self.data.SPY_pct_change)
                        self.spy_rsi = self.I(identity_func, self.data.SPY_RSI_14)
                    else: # Handle case where SPY data wasn't available
                        self.spy_pct_change = pd.Series([0] * len(self.data.Close), index=self.data.index)
                        self.spy_rsi = pd.Series([50] * len(self.data.Close), index=self.data.index)

                def next(self):
                    if len(self.rsi) < 2 or pd.isna(self.rsi[-2]) or pd.isna(self.macd_line[-2]):
                        return

                    rsi_change = self.rsi[-1] - self.rsi[-2]
                    macd_change = self.macd_line[-1] - self.macd_line[-2]

                    # Assemble the feature DataFrame for the model
                    features = pd.DataFrame([[
                        self.rsi[-1], self.macd_line[-1], self.macd_hist[-1], self.macd_signal[-1],
                        self.bbl[-1], self.bbm[-1], self.bbu[-1], self.atr[-1],
                        self.stochk[-1], self.stochd[-1], self.obv[-1], self.adx[-1], self.willr[-1],
                        rsi_change, macd_change,
                        ### TIER 1: Add market features to the prediction input ###
                        self.spy_pct_change[-1], self.spy_rsi[-1]
                    ]], columns=[
                        'RSI_14', 'MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9',
                        'BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0', 'ATRr_14',
                        'STOCHk_14_3_3', 'STOCHd_14_3_3', 'OBV', 'ADX_14', 'WILLR_14',
                        'RSI_change_1d', 'MACD_change_1d',
                        'SPY_pct_change', 'SPY_RSI_14'
                    ])
                    
                    if features.isnull().values.any():
                        return
                        
                    prediction = model.predict(features)[0]

                    ### TIER 1: Update trading logic for 3-class signals ###
                    # Prediction: 2=Buy, 1=Hold, 0=Sell
                    if prediction == 2 and not self.position:
                        self.buy()
                    elif prediction == 0 and self.position:
                        self.position.close()

            df = pd.read_csv(f"stock_data/{ticker}.csv", index_col="Date", parse_dates=True)
            
            ### TIER 1: Join market data BEFORE finding the first valid index ###
            if market_features is not None:
                df = df.join(market_features)

            temp_df = df.copy()
            temp_df.ta.rsi(append=True); temp_df.ta.macd(append=True); temp_df.ta.bbands(append=True); temp_df.ta.atr(append=True)
            temp_df.ta.stoch(append=True); temp_df.ta.obv(append=True); temp_df.ta.adx(append=True); temp_df.ta.willr(append=True)
            temp_df['RSI_change_1d'] = temp_df['RSI_14'].diff()
            temp_df['MACD_change_1d'] = temp_df['MACD_12_26_9'].diff()
            first_valid_index = temp_df.dropna().index[0]
            test_df = df.loc[first_valid_index:]

            bt = Backtest(test_df, AiStrategy, cash=10000, commission=.002)
            stats = bt.run()
            
            stats['Ticker'] = ticker
            all_stats.append(stats)

        except FileNotFoundError:
            print(f"Data file for {ticker} not found, skipping.")
        except Exception as e:
            print(f"An error occurred while backtesting {ticker}: {e}")
    
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
    
    print("\n\n--- Automated Backtest Summary (Enhanced Features) ---")
    print(summary_df)
    summary_df.to_csv("backtest_summary.csv")
    print("\nSummary saved to backtest_summary.csv")

if __name__ == "__main__":
    run_all_backtests()