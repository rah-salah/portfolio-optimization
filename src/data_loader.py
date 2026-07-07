import yfinance as yf
import pandas as pd
import os
import time

TICKERS = ['TSLA', 'BND', 'SPY']
START = '2015-01-01'
END = '2026-06-30'

def fetch_data(tickers, start, end, max_retries=3, retry_delay=5):
    print(f'Downloading data for {tickers}...')
    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            data = yf.download(tickers, start=start, end=end, auto_adjust=True, threads=False)
            close = data['Close']
            missing = [t for t in tickers if t not in close.columns]
            if missing:
                raise ValueError(f'Missing columns after download: {missing}')
            # IMPORTANT: yfinance returns multi-ticker columns sorted alphabetically,
            # not in the order `tickers` was passed. Select by actual column label
            # (already correct from yfinance) instead of blindly reassigning .columns,
            # which previously caused TSLA/BND/SPY data to be mislabeled.
            close = close[tickers]
            close.index = pd.to_datetime(close.index)
            close = close.dropna()
            if close.empty:
                raise ValueError('Downloaded data is empty after dropna()')
            print(f'Downloaded {len(close)} rows from {close.index[0].date()} to {close.index[-1].date()}')
            return close
        except Exception as e:
            last_error = e
            print(f'Attempt {attempt}/{max_retries} failed: {e}')
            if attempt < max_retries:
                print(f'Retrying in {retry_delay}s...')
                time.sleep(retry_delay)
    raise RuntimeError(f'Failed to download data after {max_retries} attempts: {last_error}')

if __name__ == '__main__':
    os.makedirs('data/processed', exist_ok=True)
    df = fetch_data(TICKERS, START, END)
    df.to_csv('data/processed/prices.csv')
    print(df.tail())
    print('Saved to data/processed/prices.csv')
