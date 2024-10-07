import argparse
from datetime import datetime, timedelta
import mplfinance as mpf
import pandas as pd
import yfinance as yf
import talib

# Function to validate date format
def valid_date(s):
    try:
        return datetime.strptime(s, "%Y-%m-%d")
    except ValueError:
        raise argparse.ArgumentTypeError(f"Not a valid date: '{s}'. Use YYYY-MM-DD format.")

# Function to fetch stock data
def fetch_stock_data(stock, start_date, end_date):
    data = yf.download(stock, start=start_date, end=end_date)
    if data.empty:
        raise ValueError(f"No data found for stock: {stock} between {start_date} and {end_date}")
    return data

# Function to plot the stock data
def plot_stock_data(stock, data):
    ap = [mpf.make_addplot(data['RSI-14'], panel=2, color='blue', ylabel='RSI-14')]
    #mpf.plot(data, type='candle', volume=True, title=f"{stock} Stock Price", style='yahoo')
    mpf.plot(data, type='candle', volume=True, title=f"{stock} Stock Price", style='yahoo', addplot=ap, panel_ratios=(3,1))


def add_rsi_to_data(data):
    data['RSI-14'] = talib.RSI(data['Adj Close'], timeperiod=14)

# Main function
def main():
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description="Stock data fetcher and plotter")
    parser.add_argument("--stock", required=True, type=str, help="Stock symbol (e.g. AAPL, TSLA)")
    parser.add_argument("--start_date", type=valid_date, help="Start date (format: YYYY-MM-DD)")
    parser.add_argument("--end_date", type=valid_date, help="End date (format: YYYY-MM-DD)")

    # Parse the arguments
    args = parser.parse_args()

    # If end_date is not provided, use today's date
    end_date = args.end_date if args.end_date else datetime.today()

    # If start_date is not provided, use one year ago from end_date
    start_date = args.start_date if args.start_date else end_date - timedelta(days=365)

    # Fetch and plot stock data
    try:
        data = fetch_stock_data(args.stock, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
        add_rsi_to_data(data)
        plot_stock_data(args.stock, data)
    except ValueError as e:
        print(e)




# Entry point
if __name__ == "__main__":
    main()
