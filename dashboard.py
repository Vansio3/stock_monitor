# dashboard.py (Lightweight Viewer Version)

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os

# --- Page Config ---
st.set_page_config(page_title="AI Quant-Desk", layout="wide")

# --- Caching Functions (Now much simpler) ---
@st.cache_data
def load_csv(filename):
    """Generic function to load any CSV file from the project."""
    if os.path.exists(filename):
        return pd.read_csv(filename)
    return None

# --- Main App Logic ---
st.title("AI Market Overview")

# 1. Load all necessary data from pre-generated CSVs
all_predictions_df = load_csv("latest_predictions.csv")
summary_df = load_csv("backtest_summary.csv")
available_tickers = sorted(all_predictions_df['Ticker'].unique()) if all_predictions_df is not None else []

if 'selected_ticker' not in st.session_state:
    st.session_state.selected_ticker = available_tickers[0] if available_tickers else None

# 2. Display the Market Overview Header
st.write("Click on a ticker to see detailed analysis. 🟢 = BUY Signal, 🟡 = HOLD/SELL Signal.")

# 3. Create the dynamic ticker grid
num_columns = 8
cols = st.columns(num_columns)
if all_predictions_df is not None:
    for i, row in all_predictions_df.iterrows():
        with cols[i % num_columns]:
            signal_emoji = "🟢" if row['Signal'] == 1 else "🟡"
            if st.button(f"{signal_emoji} {row['Ticker']}", key=row['Ticker'], use_container_width=True):
                st.session_state.selected_ticker = row['Ticker']

# 4. Display the Detailed Analysis for the selected ticker
st.divider()
selected_ticker = st.session_state.selected_ticker

if selected_ticker:
    st.header(f"Detailed Analysis: {selected_ticker}")
    
    # Load the specific data for the selected ticker
    df = pd.read_csv(f"stock_data/{selected_ticker}.csv", index_col="Date", parse_dates=True)
    
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("AI Prediction")
        if all_predictions_df is not None:
            pred_data = all_predictions_df[all_predictions_df['Ticker'] == selected_ticker].iloc[0]
            if pred_data['Signal'] == 1:
                st.success("**Signal: BUY**")
            else:
                st.warning("**Signal: HOLD / SELL**")
            st.metric(label="Model Confidence", value=f"{pred_data['Confidence']:.2%}")

        st.subheader("Historical Strategy Performance")
        if summary_df is not None:
            summary_df.set_index('Ticker', inplace=True)
            if selected_ticker in summary_df.index:
                display_stats = summary_df.loc[[selected_ticker]].T.rename(columns={selected_ticker: "Value"})
                st.table(display_stats.style.format("{:.2f}"))

    with col2:
        st.subheader("Price History")
        fig = go.Figure(data=[go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name=selected_ticker)])
        fig.update_layout(xaxis_rangeslider_visible=False, height=500)
        st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("No prediction data found. Please run '6_generate_predictions.py' locally.")