import yfinance as yf
import pandas as pd

def fetch_stock_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Fetch historical stock data for a given ticker between start and end dates.
    Args:
        ticker (str): Stock ticker symbol (e.g., 'AAPL').
        start (str): Start date in 'YYYY-MM-DD' format.
        end (str): End date in 'YYYY-MM-DD' format.
    Returns:
        pd.DataFrame: DataFrame with historical stock data.
    """
    try:
        data = yf.download(ticker, start=start, end=end)
        if data.empty:
            raise ValueError(f"No data found for ticker {ticker}.")
        return data
    except Exception as e:
        print(f"Error fetching data: {e}")
        return pd.DataFrame() 