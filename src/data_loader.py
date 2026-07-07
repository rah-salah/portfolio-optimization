import yfinance as yf
import pandas as pd
import os

TICKERS = ['TSLA', 'BND', 'SPY']
START = '2015-01-01'
END = '2026-06-30'

def fetch_data(tickers, start, end):
    print(f'Downloading data for {tickers}...')
    data = yf.download(tickers, start=start, end=end, auto_adjust=True)
    close = data['Close']
    # IMPORTANT: yfinance returns multi-ticker columns sorted alphabetically,
    # not in the order `tickers` was passed. Select by actual column label
    # (already correct from yfinance) instead of blindly reassigning .columns,
    # which previously caused TSLA/BND/SPY data to be mislabeled.
    close = close[tickers]
    close.index = pd.to_datetime(close.index)
    close = close.dropna()
    print(f'Downloaded {len(close)} rows from {close.index[0].date()} to {close.index[-1].date()}')
    return close

if __name__ == '__main__':
    os.makedirs('data/processed', exist_ok=True)
    df = fetch_data(TICKERS, START, END)
    df.to_csv('data/processed/prices.csv')
    print(df.tail())
    print('Saved to data/processed/prices.csv')
