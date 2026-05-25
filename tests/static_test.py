import os
import pandas as pd
import yfinance as yf

from datetime import datetime, timedelta
from engines.LiveTrader import LiveTrader
from ai import assert_dir, dataframe_add_statistics
from config import DUMP_DIR

def fetch_latest_mock_candle(df, index):
    row = df.iloc[index]

    return {
        'Open': row['Open'],
        'High': row['High'],
        'Low': row['Low'],
        'Close': row['Close'],
        'Volume': row['Volume']
    }

def d_last_12_to_6_days_tick(tick: str) -> str:
    dates = {
        'start': (datetime.now() - timedelta(days=12)).strftime('%Y-%m-%d'),
        'end': (datetime.now() - timedelta(days=6)).strftime('%Y-%m-%d')
    }
    assert_dir(DUMP_DIR)
    dump_path = os.path.join(DUMP_DIR, f'{tick}_4d.csv')

    df = yf.download(tick, start=dates['start'], end=dates['end'], interval='5m', multi_level_index=False)
    df.to_csv(dump_path)

    return dump_path

def d_last_6_days_tick(tick: str) -> str:
    dates = {
        'start': (datetime.now() - timedelta(days=6)).strftime('%Y-%m-%d'),
        'end': (datetime.now() - timedelta(days=3)).strftime('%Y-%m-%d')
    }
    assert_dir(DUMP_DIR)
    dump_path = os.path.join(DUMP_DIR, f'{tick}_2d.csv')

    df = yf.download(tick, start=dates['start'], end=dates['end'], interval='5m', multi_level_index=False)
    df.to_csv(dump_path)

    return dump_path

def static_test(model_name, tick: str):
    trader = LiveTrader(
        model_name,
        symbol=tick
    )
    trader.df = pd.DataFrame()

    dump_path = d_last_12_to_6_days_tick(tick)
    dump_df = pd.read_csv(dump_path, parse_dates=True, index_col=0)

    trader.df = pd.concat([
        trader.df,
        dump_df
    ])

    os.remove(dump_path)
    dump_path = d_last_6_days_tick(tick)
    dump_df = pd.read_csv(dump_path, parse_dates=True, index_col=0)

    for i in range(len(dump_df)):
        candle = fetch_latest_mock_candle(dump_df, i)

        trader.on_new_candle(candle)

