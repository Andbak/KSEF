import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import datetime as dt
import matplotlib.pyplot as plt
import cvxpy as cp
from sklearn.linear_model import LinearRegression
from sklearn.covariance import LedoitWolf

# -------------------------------------------
# 1. Title
st.title(
    "Portfolio Optimization (MVP & Tangency, CAPM + Shrinkage)"
)

# -------------------------------------------
# 2. User Inputs
st.sidebar.header("User Input Parameters")

tickers_input = st.sidebar.text_input(
    "Enter stock tickers separated by commas", "AAPL, MSFT, TSLA, AMZN, NVDA"
)
tickers = [ticker.strip().upper() for ticker in tickers_input.split(",")]

years = st.sidebar.slider("Years of historical data", 1, 10, 5)
risk_free_rate_input = (
    st.sidebar.number_input(
        "Risk-Free Rate (%)", min_value=0.0, max_value=10.0, value=2.0
    )
    / 100
)
market_return_input = (
    st.sidebar.number_input(
        "Expected Market Return (%)", min_value=0.0, max_value=20.0, value=8.0
    )
    / 100
)
max_weight_slider = st.sidebar.slider("Max weight per stock (%)", 5, 100, 30)
min_weight_slider = st.sidebar.slider("Min weight per stock (%)", 0, 10, 0)
max_weight = max_weight_slider / 100
min_weight = min_weight_slider / 100

end_date = dt.datetime.today()
start_date = end_date - dt.timedelta(days=years * 365)

# -------------------------------------------
# 3. Download Stock and Market Data
market_index = "^GSPC"  # S&P500 Index

data = yf.download(tickers, start=start_date, end=end_date)["Close"]
market_data = yf.download(market_index, start=start_date, end=end_date)["Close"]

if data.empty or market_data.empty:
    st.error("No data fetched. Please check your tickers or internet connection.")
    st.stop()

returns = data.pct_change().dropna()
market_returns = market_data.pct_change().dropna()

# -------------------------------------------
# 4. Estimate Betas (for CAPM Expected Returns)
betas = []
for ticker in tickers:
    model = LinearRegression()
    model.fit(
        market_returns.values.reshape(-1, 1), returns[ticker].values.reshape(-1, 1)
    )
    betas.append(model.coef_[0][0])

# CAPM Expected Returns
capm_returns = risk_free_rate_input + np.array(betas) * (
    market_return_input - risk_free_rate_input
)

# Historical mean returns
historical_mean_returns = returns.mean() * 252

# Blend CAPM and historical returns (50/50)
expected_returns = 0.5 * capm_returns + 0.5 * historical_mean_returns.values

# Ledoit-Wolf shrinkage covariance matrix
lw = LedoitWolf()
cov_matrix = lw.fit(returns).covariance_ * 252

# -------------------------------------------
# 5. Optimization Functions using CVXPY


def solve_mvp(cov_matrix, min_weight, max_weight):
    n = cov_matrix.shape[0]
    w = cp.Variable(n)
    objective = cp.Minimize(cp.quad_form(w, cov_matrix))
    constraints = [cp.sum(w) == 1, w >= min_weight, w <= max_weight]
    problem = cp.Problem(objective, constraints)
    problem.solve()
    return w.value


def solve_tangency(
    expected_returns, cov_matrix, risk_free_rate, min_weight, max_weight
):
    n = cov_matrix.shape[0]
    w = cp.Variable(n)
    excess_returns = expected_returns - risk_free_rate
    objective = cp.Maximize(excess_returns @ w)
    constraints = [
        cp.sum(w) == 1,
        w >= min_weight,
        w <= max_weight,
        cp.quad_form(w, cov_matrix) <= 1,
    ]
    problem = cp.Problem(objective, constraints)
    problem.solve()
    return w.value


# -------------------------------------------
# 6. Optimize and Display
st.header("Optimization Results")

if st.button("Optimize Portfolio"):
    try:
        mvp_weights = solve_mvp(cov_matrix, min_weight, max_weight)
        tangency_weights = solve_tangency(
            expected_returns, cov_matrix, risk_free_rate_input, min_weight, max_weight
        )

        # MVP
        mvp_ret = np.dot(mvp_weights, expected_returns)
        mvp_vol = np.sqrt(np.dot(mvp_weights.T, np.dot(cov_matrix, mvp_weights)))
        mvp_sharpe = (mvp_ret - risk_free_rate_input) / mvp_vol

        # Tangency
        tangency_ret = np.dot(tangency_weights, expected_returns)
        tangency_vol = np.sqrt(
            np.dot(tangency_weights.T, np.dot(cov_matrix, tangency_weights))
        )
        tangency_sharpe = (tangency_ret - risk_free_rate_input) / tangency_vol

        # --- Display MVP
        st.subheader("Minimum Variance Portfolio (MVP)")
        mvp_df = pd.DataFrame({"Ticker": tickers, "Weight": mvp_weights})
        mvp_df["Weight (%)"] = mvp_df["Weight"] * 100
        st.dataframe(
            mvp_df[["Ticker", "Weight (%)"]].style.format({"Weight (%)": "{:.2f}%"})
        )

        st.write(f"Expected Return: **{mvp_ret:.2%}**")
        st.write(f"Volatility: **{mvp_vol:.2%}**")
        st.write(f"Sharpe Ratio: **{mvp_sharpe:.2f}**")

        # --- Display Tangency Portfolio
        st.subheader("Tangency Portfolio (Max Sharpe)")
        tangency_df = pd.DataFrame({"Ticker": tickers, "Weight": tangency_weights})
        tangency_df["Weight (%)"] = tangency_df["Weight"] * 100
        st.dataframe(
            tangency_df[["Ticker", "Weight (%)"]].style.format(
                {"Weight (%)": "{:.2f}%"}
            )
        )

        st.write(f"Expected Return: **{tangency_ret:.2%}**")
        st.write(f"Volatility: **{tangency_vol:.2%}**")
        st.write(f"Sharpe Ratio: **{tangency_sharpe:.2f}**")

        # --- Pie Charts
        st.subheader("Portfolio Allocation Charts")

        col1, col2 = st.columns(2)
        with col1:
            st.write("Minimum Variance Portfolio")
            fig_mvp, ax_mvp = plt.subplots()
            ax_mvp.pie(
                mvp_df["Weight"],
                labels=mvp_df["Ticker"],
                autopct="%1.1f%%",
                startangle=90,
            )
            ax_mvp.axis("equal")
            st.pyplot(fig_mvp)

        with col2:
            st.write("Tangency Portfolio")
            fig_tang, ax_tang = plt.subplots()
            ax_tang.pie(
                tangency_df["Weight"],
                labels=tangency_df["Ticker"],
                autopct="%1.1f%%",
                startangle=90,
            )
            ax_tang.axis("equal")
            st.pyplot(fig_tang)

        # --- Download Buttons
        st.subheader("Download Optimized Portfolios")

        mvp_csv = mvp_df[["Ticker", "Weight"]].to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download MVP Portfolio as CSV",
            data=mvp_csv,
            file_name="mvp_portfolio.csv",
            mime="text/csv",
        )

        tangency_csv = (
            tangency_df[["Ticker", "Weight"]].to_csv(index=False).encode("utf-8")
        )
        st.download_button(
            label="Download Tangency Portfolio as CSV",
            data=tangency_csv,
            file_name="tangency_portfolio.csv",
            mime="text/csv",
        )

    except Exception as e:
        st.error(f"Optimization failed: {str(e)}")

# -------------------------------------------
# 7. Show Raw Data
st.header("Historical Stock Price Data")
st.dataframe(data)
