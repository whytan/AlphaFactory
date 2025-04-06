import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from jinja2 import Template
import datetime
import os

st.set_page_config(page_title="AlphaFactory", layout="wide")
st.title("📈 AlphaFactory – Factor Investing Strategy Backtester")

# Sidebar inputs
with st.sidebar:
    st.header("📊 Strategy Configuration")
    ticker_input = st.text_input("Enter stock tickers (comma-separated):", "")
    start_date = st.date_input("Start Date", datetime.date(2018, 1, 1))
    end_date = st.date_input("End Date", datetime.date(2023, 12, 31))
    strategy = st.selectbox("Select Strategy", ["Momentum", "Low Volatility"])
    top_n = st.slider("Select Top N Stocks", 1, 10, 3)
    run_button = st.button("🚀 Run Backtest")

# Main logic only runs if button is clicked
if run_button:
    tickers = [ticker.strip().upper() for ticker in ticker_input.split(",") if ticker.strip()]
    if len(tickers) == 0:
        st.warning("Please enter at least one valid ticker.")
        st.stop()

    try:
        # Download price data
        data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=False)["Adj Close"]
        if data.empty:
            st.error("No data found. Please check tickers or dates.")
            st.stop()

        daily_returns = data.pct_change().dropna()

        # Score calculation
        window = 30
        if strategy == "Momentum":
            scores = daily_returns.rolling(window).mean()
        else:
            scores = -daily_returns.rolling(window).std()

        scores = scores.dropna()
        portfolio_returns = []

        for date in scores.index:
            top_stocks = scores.loc[date].nlargest(top_n).index
            daily_ret = daily_returns.loc[date, top_stocks].mean()
            portfolio_returns.append((date, daily_ret))

        portfolio_returns = pd.Series(dict(portfolio_returns))
        cumulative_returns = (1 + portfolio_returns).cumprod()

        st.subheader("📈 Strategy Cumulative Returns")
        fig, ax = plt.subplots(figsize=(10, 4))
        cumulative_returns.plot(ax=ax, label="Strategy")
        ax.set_ylabel("Growth of $1")
        ax.legend()
        st.pyplot(fig)

        sharpe_ratio = portfolio_returns.mean() / portfolio_returns.std() * np.sqrt(252)
        total_return = cumulative_returns.iloc[-1] - 1
        st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
        st.metric("Total Return", f"{total_return:.2%}")

        # Benchmark comparison
        benchmark_data = yf.download("SPY", start=start_date, end=end_date, auto_adjust=False)["Adj Close"]
        benchmark_returns = benchmark_data.pct_change().dropna()
        benchmark_cum = (1 + benchmark_returns).cumprod()
        benchmark_returns = benchmark_returns.loc[portfolio_returns.index]
        benchmark_cum = benchmark_cum.loc[portfolio_returns.index]
        benchmark_sharpe = benchmark_returns.mean() / benchmark_returns.std() * np.sqrt(252)
        benchmark_total = benchmark_cum.iloc[-1] - 1

        st.subheader("📊 Strategy vs Benchmark")
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        cumulative_returns.plot(ax=ax2, label="Strategy")
        benchmark_cum.plot(ax=ax2, label="SPY", linestyle="--")
        ax2.set_ylabel("Growth of $1")
        ax2.legend()
        st.pyplot(fig2)

        # Generate HTML report
        with open("report_template.html") as file:
            template = Template(file.read())

        html_out = template.render(
            tickers=", ".join(tickers),
            start_date=start_date,
            end_date=end_date,
            strategy=strategy,
            top_n=top_n,
            sharpe_ratio=f"{sharpe_ratio:.2f}",
            total_return=f"{total_return:.2%}",
            benchmark_sharpe=f"{benchmark_sharpe.item():.2f}" if hasattr(benchmark_sharpe, 'item') else f"{benchmark_sharpe:.2f}",
            benchmark_total=f"{benchmark_total.item():.2%}" if hasattr(benchmark_total, 'item') else f"{benchmark_total:.2%}",
            summary="✅ Your strategy outperformed the S&P 500!" if total_return > benchmark_total.item() if hasattr(benchmark_total, 'item') else benchmark_total else "⚠️ Your strategy underperformed the S&P 500."
        )

        html_path = "AlphaFactory_Report.html"
        with open(html_path, "w") as out:
            out.write(html_out)

        with open(html_path, "rb") as file:
            st.download_button("📄 Download Strategy Report (HTML)", file, file_name="AlphaFactory_Report.html")

    except Exception as e:
        st.error(f"❌ An error occurred: {e}")
