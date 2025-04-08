import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.linear_model import LinearRegression
from pypfopt import (
    EfficientFrontier,
    risk_models,
    expected_returns,
    objective_functions,
)
import matplotlib.pyplot as plt


# -------------------------------------------
# 1. Helper Function: Normalize non-zero weights
def normalize_weights(weights_dict):
    weights = pd.Series(weights_dict)
    non_zero_weights = weights[weights > 0]

    if non_zero_weights.sum() < 0.999:  # Allow some tolerance
        non_zero_weights = non_zero_weights / non_zero_weights.sum()

    final_weights = pd.concat([non_zero_weights, weights[weights == 0]])
    return final_weights.to_dict()


# -------------------------------------------
# 2. Streamlit Title
st.title("🇳🇴 Norwegian Stocks Portfolio Optimization (MVP + Tangency, CAPM Corrected)")

# -------------------------------------------
# 3. User Inputs
st.sidebar.header("Input Parameters")

tickers_input = st.sidebar.text_input(
    "Enter Norwegian stock tickers separated by commas (e.g., EQNR.OL, DNB.OL, AKRBP.OL)",
    "EQNR.OL, DNB.OL, AKRBP.OL",
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
        "Expected Market Return (%)", min_value=0.0, max_value=20.0, value=7.0
    )
    / 100
)
max_weight_slider = st.sidebar.slider("Max weight per stock (%)", 5, 100, 30)
max_weight = max_weight_slider / 100

end_date = pd.to_datetime("today")
start_date = end_date - pd.DateOffset(years=years)

# -------------------------------------------
# 4. Download Data
market_index = "^OSEAX"  # Oslo All-Share Index

try:
    data = yf.download(tickers, start=start_date, end=end_date)["Close"]
    market_data = yf.download(market_index, start=start_date, end=end_date)["Close"]

    returns = data.pct_change().dropna()
    market_returns = market_data.pct_change().dropna()

except Exception as e:
    st.error(f"Data fetching failed: {str(e)}")
    st.stop()

if data.empty or market_data.empty:
    st.error("No data available. Please check your tickers and try again.")
    st.stop()

# Check max_weight feasibility
if max_weight * len(tickers) < 1:
    st.error(
        "Max weight is too low for the number of selected stocks. Please increase max weight or add more stocks."
    )
    st.stop()

# -------------------------------------------
# 5. Correct CAPM Beta Estimation
betas = []
for ticker in tickers:
    combined = pd.concat([returns[ticker], market_returns], axis=1).dropna()
    X = combined.iloc[:, 1].values.reshape(-1, 1)  # Market returns
    y = combined.iloc[:, 0].values.reshape(-1, 1)  # Stock returns
    model = LinearRegression()
    model.fit(X, y)
    betas.append(model.coef_[0][0])

capm_returns = risk_free_rate_input + np.array(betas) * (
    market_return_input - risk_free_rate_input
)

# Use pure CAPM returns
expected_ret = pd.Series(capm_returns, index=tickers)

# Covariance matrix with Ledoit-Wolf shrinkage
cov_matrix = risk_models.CovarianceShrinkage(data).ledoit_wolf()

# -------------------------------------------
# 6. Optimization with PyPortfolioOpt and L2 Regularization

# MVP
ef_mvp = EfficientFrontier(expected_ret, cov_matrix, weight_bounds=(0, max_weight))
ef_mvp.add_objective(objective_functions.L2_reg, gamma=0.4)
mvp_weights = ef_mvp.min_volatility()
mvp_cleaned = normalize_weights(ef_mvp.clean_weights())

# Tangency
ef_tangency = EfficientFrontier(expected_ret, cov_matrix, weight_bounds=(0, max_weight))
ef_tangency.add_objective(objective_functions.L2_reg, gamma=0.4)
tangency_weights = ef_tangency.max_sharpe(risk_free_rate=risk_free_rate_input)
tangency_cleaned = normalize_weights(ef_tangency.clean_weights())

# Portfolio performance
mvp_perf = ef_mvp.portfolio_performance(
    verbose=False, risk_free_rate=risk_free_rate_input
)
tangency_perf = ef_tangency.portfolio_performance(
    verbose=False, risk_free_rate=risk_free_rate_input
)

# -------------------------------------------
# 7. Display Results
st.header("Optimized Portfolios")

# MVP
st.subheader("Minimum Variance Portfolio (MVP)")
mvp_df = pd.DataFrame(mvp_cleaned.items(), columns=["Ticker", "Weight"])
mvp_df["Weight (%)"] = mvp_df["Weight"] * 100
st.dataframe(mvp_df.style.format({"Weight (%)": "{:.2f}%"}))

st.write(f"📈 Expected Return: **{mvp_perf[0]:.2%}**")
st.write(f"📉 Volatility: **{mvp_perf[1]:.2%}**")
st.write(f"⚡ Sharpe Ratio: **{mvp_perf[2]:.2f}**")

# Tangency
st.subheader("Tangency Portfolio (Max Sharpe)")
tangency_df = pd.DataFrame(tangency_cleaned.items(), columns=["Ticker", "Weight"])
tangency_df["Weight (%)"] = tangency_df["Weight"] * 100
st.dataframe(tangency_df.style.format({"Weight (%)": "{:.2f}%"}))

st.write(f"📈 Expected Return: **{tangency_perf[0]:.2%}**")
st.write(f"📉 Volatility: **{tangency_perf[1]:.2%}**")
st.write(f"⚡ Sharpe Ratio: **{tangency_perf[2]:.2f}**")

# -------------------------------------------
# 8. Plot Pie Charts
st.header("Portfolio Allocations")

col1, col2 = st.columns(2)

with col1:
    st.write("MVP Allocation")
    fig1, ax1 = plt.subplots()
    ax1.pie(mvp_df["Weight"], labels=mvp_df["Ticker"], autopct="%1.1f%%", startangle=90)
    ax1.axis("equal")
    st.pyplot(fig1)

with col2:
    st.write("Tangency Allocation")
    fig2, ax2 = plt.subplots()
    ax2.pie(
        tangency_df["Weight"],
        labels=tangency_df["Ticker"],
        autopct="%1.1f%%",
        startangle=90,
    )
    ax2.axis("equal")
    st.pyplot(fig2)

# -------------------------------------------
# 9. Download Portfolios
st.header("Download Portfolios")

mvp_csv = mvp_df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Download MVP Portfolio as CSV",
    data=mvp_csv,
    file_name="mvp_portfolio.csv",
    mime="text/csv",
)

tangency_csv = tangency_df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Download Tangency Portfolio as CSV",
    data=tangency_csv,
    file_name="tangency_portfolio.csv",
    mime="text/csv",
)
