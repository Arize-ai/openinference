# filename: ytd_gain.py
import yfinance as yf

# Define the stock tickers
tickers = ['META', 'TSLA']

# Define the start and end dates
start_date = '2025-01-01'
end_date = '2025-01-17'

# Fetch the stock data
data = yf.download(tickers, start=start_date, end=end_date)

# Get the opening price at the start of the year and the closing price on the current date
meta_start_price = data['Open']['META'].iloc[0]
meta_end_price = data['Close']['META'].iloc[-1]
tesla_start_price = data['Open']['TSLA'].iloc[0]
tesla_end_price = data['Close']['TSLA'].iloc[-1]

# Calculate the year-to-date gain
meta_ytd_gain = ((meta_end_price - meta_start_price) / meta_start_price) * 100
tesla_ytd_gain = ((tesla_end_price - tesla_start_price) / tesla_start_price) * 100

# Print the results
print(f"META Year-to-Date Gain: {meta_ytd_gain:.2f}%")
print(f"TESLA Year-to-Date Gain: {tesla_ytd_gain:.2f}%")