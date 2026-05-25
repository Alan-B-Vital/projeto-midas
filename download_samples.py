import yfinance as yf
import os
from multiprocessing import Pool, cpu_count

from datetime import datetime, timedelta
from config import TRAIN_SAMPLE_DIR, TEST_SAMPLE_DIR

def assert_dir(dir) -> None:
    if not os.path.exists(dir):
        os.makedirs(dir)

def d_5m_tick(tick: str):
    date_train = {
        'start': (datetime.now() - timedelta(days=59)).strftime('%Y-%m-%d'),
        'end': (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    }
    date_test= {
        'start': (datetime.now() - timedelta(days=29)).strftime('%Y-%m-%d'),
        'end': datetime.now().strftime('%Y-%m-%d')
    }

    df = yf.download(tick, start=date_train['start'], end=date_train['end'], interval='5m', multi_level_index=False)
    df.to_csv(f'{TRAIN_SAMPLE_DIR}/{tick}_1M_5m.csv')
    df = yf.download(tick, start=date_test['start'], end=date_test['end'], interval='5m', multi_level_index=False)
    df.to_csv(f'{TEST_SAMPLE_DIR}/{tick}_1M_5m.csv')

def download_1M_5m(ticks: tuple) -> None:
    assert_dir(TRAIN_SAMPLE_DIR)
    assert_dir(TEST_SAMPLE_DIR)

    pool = Pool(processes=(cpu_count() - 1))

    pool.map(d_5m_tick, ticks)

    pool.close()
    pool.join()

def d_1d_tick(tick: str):
    date_train = {
        'start': (datetime.now() - timedelta(days=365 * 2)).strftime('%Y-%m-%d'),
        'end': (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    }
    date_test= {
        'start': (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d'),
        'end': datetime.now().strftime('%Y-%m-%d')
    }

    df = yf.download(tick, start=date_train['start'], end=date_train['end'], interval='1d', multi_level_index=False)
    df.to_csv(f'{TRAIN_SAMPLE_DIR}/{tick}_1Y_1D.csv')
    df = yf.download(tick, start=date_test['start'], end=date_test['end'], interval='1d', multi_level_index=False)
    df.to_csv(f'{TEST_SAMPLE_DIR}/{tick}_1Y_1D.csv')

def download_1Y_1D(ticks: tuple) -> None:
    assert_dir(TRAIN_SAMPLE_DIR)
    assert_dir(TEST_SAMPLE_DIR)

    pool = Pool(processes=(cpu_count() - 1))

    pool.map(d_1d_tick, ticks)

    pool.close()
    pool.join()
