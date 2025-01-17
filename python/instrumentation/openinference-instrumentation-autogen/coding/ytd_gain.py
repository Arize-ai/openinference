# filename: ytd_gain.py

import datetime
import yfinance as yf

# Step 1: Get today's date
today = datetime.date.today()
print(f"Today's date is: {today}")

# Step 2: Fetch stock data for META and TESLA
start_of_year = datetime.date(today.year, 1, 1)

# Fetch data using yfinance
meta_data = yf.download("META", start=start_of_year, end=today)
tesla_data = yf.download("TSLA", start=start_of_year, end=today)

# Debug: Print the first few rows of the data to understand its structure
print("META data:")
print(meta_data.head())
print("\nTESLA data:")
print(tesla_data.head())

# Step 3: Calculate year-to-date gain
def calculate_ytd_gain(data, ticker):
    if data.empty:
        return None
    # Access the specific value using the ticker symbol
    start_price = data['Close'].iloc[0][ticker]
    end_price = data['Close'].iloc[-1][ticker]
    
    # Debug: Print the specific values
    print(f"Start price: {start_price}, End price: {end_price}")
    
    ytd_gain = ((end_price - start_price) / start_price) * 100
    return ytd_gain

meta_ytd_gain = calculate_ytd_gain(meta_data, 'META')
tesla_ytd_gain = calculate_ytd_gain(tesla_data, 'TSLA')

if meta_ytd_gain is not None:
    print(f"Year-to-date gain for META: {meta_ytd_gain:.2f}%")
else:
    print("No data available for META.")

if tesla_ytd_gain is not None:
    print(f"Year-to-date gain for TESLA: {tesla_ytd_gain:.2f}%")
else:
    print("No data available for TESLA.")