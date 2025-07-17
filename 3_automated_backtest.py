# 3_automated_backtest.py (CORRECTED, SIMPLIFIED LOGIC)

import pandas as pd
import pandas_ta as ta
from backtesting import Backtest, Strategy
import joblib
import os

# --- Main Backtesting Loop ---
def run_all_backtests():
    """Finds all models, runs a backtest for each, and prints a summary."""
    
    # 1. Get the list of tickers that have a trained model
    try:
        model_files = os.listdir('models')
        tickers = sorted([f.split('_model.pkl')[0] for f in model_files if f.endswith('.pkl')])
    except FileNotFoundError:
        print("Error: 'models' directory not found. Please train models first using '2_model_trainer.py'.")
        return

    if not tickers:
        print("No models found in the 'models' directory.")
        return

    all_stats = []
    print(f"Found models for: {', '.join(tickers)}. Running backtests...")

    # 2. Loop through each ticker
    for ticker in tickers:
        print(f"--- Backtesting for {ticker} ---")
        try:
            # --- Load the specific model for this ticker FIRST ---
            model = joblib.load(f"models/{ticker}_model.pkl")

            # --- Define the Strategy CLASS INSIDE the loop ---
            # This allows it to "capture" the currently loaded model.
            class AiStrategy(Strategy):
                def init(self):
                    close_series = pd.Series(self.data.Close)
                    self.rsi = self.I(ta.rsi, close_series)
                    self.macd_line = self.I(lambda x: ta.macd(x).iloc[:, 0], close_series)
                    self.macd_hist = self.I(lambda x: ta.macd(x).iloc[:, 1], close_series)
                    self.macd_signal = self.I(lambda x: ta.macd(x).iloc[:, 2], close_series)
                    self.bbl = self.I(lambda x: ta.bbands(x, length=20).iloc[:, 0], close_series) 
                    self.bbm = self.I(lambda x: ta.bbands(x, length=20).iloc[:, 1], close_series)
                    self.bbu = self.I(lambda x: ta.bbands(x, length=20).iloc[:, 2], close_series)
                    self.atr = self.I(lambda high, low, close: ta.atr(high=pd.Series(high), low=pd.Series(low), close=pd.Series(close)),
                                      self.data.High, self.data.Low, self.data.Close)

                def next(self):
                    latest_rsi = self.rsi[-1]
                    latest_macd_line = self.macd_line[-1]
                    latest_macd_hist = self.macd_hist[-1]
                    latest_macd_signal = self.macd_signal[-1]
                    latest_bbl = self.bbl[-1]
                    latest_bbm = self.bbm[-1]
                    latest_bbu = self.bbu[-1]
                    latest_atr = self.atr[-1]

                    features = pd.DataFrame([[
                        latest_rsi, latest_macd_line, latest_macd_hist, latest_macd_signal,
                        latest_bbl, latest_bbm, latest_bbu, latest_atr
                    ]], columns=[
                        'RSI_14', 'MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9',
                        'BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0', 'ATRr_14'
                    ])
                    
                    # The strategy uses the 'model' from the outer scope
                    prediction = model.predict(features)[0]

                    if prediction == 1 and not self.position:
                        self.buy()
                    elif prediction == 0 and self.position:
                        self.position.close()

            # Load the data for the current ticker
            df = pd.read_csv(f"stock_data/{ticker}.csv", index_col="Date", parse_dates=True)

            # Find the correct start date
            temp_df = df.copy()
            temp_df.ta.rsi(append=True); temp_df.ta.macd(append=True); temp_df.ta.bbands(length=20, append=True); temp_df.ta.atr(append=True)
            first_valid_index = temp_df.dropna().index[0]
            test_df = df.loc[first_valid_index:]

            # Run the backtest
            bt = Backtest(test_df, AiStrategy, cash=10000, commission=.002)
            stats = bt.run()
            
            stats['Ticker'] = ticker
            all_stats.append(stats)

        except FileNotFoundError:
            print(f"Data file for {ticker} not found, skipping.")
        except Exception as e:
            print(f"An error occurred while backtesting {ticker}: {e}")
    
    # 3. Create and display the summary DataFrame (same as before)
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
    
    print("\n\n--- Automated Backtest Summary ---")
    print(summary_df)

    # Save the summary to a file for the dashboard to use
    summary_df.to_csv("backtest_summary.csv")
    print("\nSummary saved to backtest_summary.csv")

if __name__ == "__main__":
    run_all_backtests()