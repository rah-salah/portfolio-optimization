import pytest
import pandas as pd
import numpy as np
import os

def test_prices_file_exists():
    assert os.path.exists('data/processed/prices.csv')

def test_prices_shape():
    df = pd.read_csv('data/processed/prices.csv', index_col=0, parse_dates=True)
    assert df.shape[1] == 3
    assert len(df) > 2000

def test_no_missing_values():
    df = pd.read_csv('data/processed/prices.csv', index_col=0, parse_dates=True)
    assert df.isnull().sum().sum() == 0

def test_correct_tickers():
    df = pd.read_csv('data/processed/prices.csv', index_col=0, parse_dates=True)
    assert set(df.columns) == {"TSLA", "BND", "SPY"}

def test_chronological_order():
    df = pd.read_csv('data/processed/prices.csv', index_col=0, parse_dates=True)
    assert df.index.is_monotonic_increasing
