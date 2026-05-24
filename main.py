import os

from download_samples import *
from backtest import *
from ai import *

mega_ticks = [
    'SPY',
    'QQQ',
    'AAPL',
    'MSFT',
    'AMZN',
    'META',
    'GOOG',
    'JPM',
    'XOM',
    'WMT'
]

TEST_SAMPLE_DIR = './samples/test'
def clean():
    if os.path.exists(mp := os.path.join(MODEL_DIR, f'{MODEL_NAME}.json')):
        os.remove(mp)

    if os.path.exists(RESULT_DIR):
        for f in os.listdir(RESULT_DIR):
            os.remove(os.path.join(RESULT_DIR, f))
        os.rmdir(RESULT_DIR)

if __name__ == '__main__':
    MODEL_NAME = 'mega'
    
    TICKS=[(T, T.lower()) for T in mega_ticks]
    download_1M_5m(TICKS)

    # MODEL_NAME = '1d'
    # download_1Y_1D()

    clean()
    train_model(MODEL_NAME)
    test_model(MODEL_NAME)

    # for s in [sample.split('.')[0] for sample in os.listdir(TRAIN_SAMPLE_DIR)]:
    #     benchmark(s)
