import streamlit as st
import pandas as pd
import yfinance as yf
import datetime as dt

st.write("""
# Stock Price App
This app retrieves stock prices from Yahoo Finance!
""")

ticker = st.text_input("Enter stock ticker", "AAPL")
years = st.slider("Years of data", 1, 5, 1)
end_date = dt.datetime.today().strftime("%Y-%m-%d")
start_date = (dt.datetime.today() - dt.timedelta(days=365 * years)).strftime("%Y-%m-%d")

data = yf.download(ticker, start=start_date, end=end_date)

st.write(f"## {ticker} Stock Price Data")
st.line_chart(data["Close"])
st.write("## Volume")
st.bar_chart(data["Volume"])

csv = st.download_button(
    label="Download data as CSV",
    data=data["Close"].to_csv().encode("utf-8"),
    file_name=f"{ticker}_stock_data.csv",
    mime="text/csv",
)

close = data["Close"]

ticker

