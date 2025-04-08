import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.linear_model import LinearRegression
from pypfopt import EfficientFrontier, risk_models, expected_returns, objective_functions, plotting
import matplotlib.pyplot as plt

# -------------------------------------------
# Helper function to normalize non-zero weights
def normalize_weights(weights_dict):
    weights = pd.Series(weights_dict)
    non_zero = weights[weights > 0]
    return (non_zero / non_zero.sum()).reindex(weights.index, fill_value=0).to_dict()

# -------------------------------------------
# App Title
st.title("Norwegian Stocks Portfolio Optimization (MVP + Tangency + Efficient Frontier)")

# -------------------------------------------
# Sidebar User Inputs
st.sidebar.header("Input Parameters")
tickers = [t.strip().upper() for t in st.sidebar.text_input(
    "Enter stock tickers (e.g., EQNR.OL, DNB.OL)", "EQNR.OL, DNB.OL, AKRBP.OL").split(",")]

years = st.sidebar.slider("Years of historical data", 1, 10, 5)
risk_free_rate = st.sidebar.number_input("Risk-Free Rate (%)", 0.0, 10.0, 2.0) / 100
market_return = st.sidebar.number_input("Expected Market Return (%)", 0.0, 20.0, 7.0) / 100
max_weight = st.sidebar.slider("Max weight per stock (%)", 5, 100, 50) / 100
gamma = st.sidebar.slider("L2 Regularization Strength (gamma)", 0.0, 10.0, 0.2, step=0.1)

# -------------------------------------------
# Data download
start_date, end_date = pd.to_datetime("today") - pd.DateOffset(years=years), pd.to_datetime("today")
market_index = "^OSEAX"

try:
    data = yf.download(tickers, start=start_date, end=end_date)["Close"]
    market_data = yf.download(market_index, start=start_date, end=end_date)["Close"]
    returns, market_returns = data.pct_change().dropna(), market_data.pct_change().dropna()
except Exception as e:
    st.error(f"Data fetching failed: {e}")
    st.stop()

if data.empty or market_data.empty:
    st.error("No data available. Please check your tickers.")
    st.stop()

if max_weight * len(tickers) < 1:
    st.error("Max weight too small for number of stocks. Increase max weight or add more stocks.")
    st.stop()

# -------------------------------------------
# CAPM Expected Returns
betas = []
for ticker in tickers:
    combined = pd.concat([returns[ticker], market_returns], axis=1).dropna()
    model = LinearRegression().fit(combined.iloc[:,1].values.reshape(-1,1), combined.iloc[:,0].values.reshape(-1,1))
    betas.append(model.coef_[0][0])

capm_returns = risk_free_rate + np.array(betas) * (market_return - risk_free_rate)
expected_ret = pd.Series(capm_returns, index=tickers)
cov_matrix = risk_models.CovarianceShrinkage(data).ledoit_wolf()

# -------------------------------------------
# Optimization with PyPortfolioOpt
def optimize_portfolio(objective_func):
    ef = EfficientFrontier(expected_ret, cov_matrix, weight_bounds=(0, max_weight))
    ef.add_objective(objective_functions.L2_reg, gamma=gamma)
    if objective_func == "min_volatility":
        weights = ef.min_volatility()
    else:
        weights = ef.max_sharpe(risk_free_rate=risk_free_rate)
    perf = ef.portfolio_performance(verbose=False, risk_free_rate=risk_free_rate)
    cleaned = normalize_weights(ef.clean_weights())
    return pd.DataFrame(cleaned.items(), columns=["Ticker", "Weight"]), perf

mvp_df, mvp_perf = optimize_portfolio("min_volatility")
tangency_df, tangency_perf = optimize_portfolio("max_sharpe")

# -------------------------------------------
# Display Results
st.header("Optimized Portfolios")

def show_portfolio(df, perf, title):
    df["Weight (%)"] = df["Weight"] * 100
    st.subheader(title)
    st.dataframe(df.style.format({"Weight (%)": "{:.2f}%"}))
    st.write(f"Expected Return: **{perf[0]:.2%}**")
    st.write(f"Volatility: **{perf[1]:.2%}**")
    st.write(f"Sharpe Ratio: **{perf[2]:.2f}**")

show_portfolio(mvp_df, mvp_perf, "Minimum Variance Portfolio (MVP)")
show_portfolio(tangency_df, tangency_perf, "Tangency Portfolio (Max Sharpe)")

# -------------------------------------------
# Pie Charts
st.header("Portfolio Allocations")

col1, col2 = st.columns(2)
with col1:
    st.write("MVP Allocation")
    fig, ax = plt.subplots()
    ax.pie(mvp_df["Weight"], labels=mvp_df["Ticker"], autopct="%1.1f%%", startangle=90)
    ax.axis("equal")
    st.pyplot(fig)

with col2:
    st.write("Tangency Allocation")
    fig, ax = plt.subplots()
    ax.pie(tangency_df["Weight"], labels=tangency_df["Ticker"], autopct="%1.1f%%", startangle=90)
    ax.axis("equal")
    st.pyplot(fig)

# -------------------------------------------
# Download buttons
st.header("Download Portfolios")

st.download_button(
    label="Download MVP Portfolio as CSV",
    data=mvp_df.to_csv(index=False).encode("utf-8"),
    file_name="mvp_portfolio.csv",
    mime="text/csv"
)

st.download_button(
    label="Download Tangency Portfolio as CSV",
    data=tangency_df.to_csv(index=False).encode("utf-8"),
    file_name="tangency_portfolio.csv",
    mime="text/csv"
)

# -------------------------------------------
# Efficient Frontier Plot
st.header("Efficient Frontier Plot")

fig, ax = plt.subplots(figsize=(8, 6))
plotting.plot_efficient_frontier(EfficientFrontier(expected_ret, cov_matrix), ax=ax, show_assets=True)

# Plot MVP and Tangency points
ax.scatter(mvp_perf[1], mvp_perf[0], marker="o", color="red", label="MVP")
ax.scatter(tangency_perf[1], tangency_perf[0], marker="*", color="gold", s=100, label="Tangency")

# Capital Market Line
x = np.linspace(0, max(mvp_perf[1], tangency_perf[1])*1.5, 100)
cml = risk_free_rate + tangency_perf[2] * x
ax.plot(x, cml, linestyle="--", color="green", label="Capital Market Line")

ax.set_title("Efficient Frontier with MVP and Tangency")
ax.set_xlabel("Volatility (Standard Deviation)")
ax.set_ylabel("Expected Return")
ax.legend()
st.pyplot(fig)
