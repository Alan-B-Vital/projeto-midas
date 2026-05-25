import os

from config import MODEL_DIR, RESULT_DIR, MODEL_NAME
from download_samples import download_1M_5m
from ai import train_model, test_model


# Stable and Large trade volumes Ticks
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

def clean_model_and_test_results():
    if os.path.exists(mp := os.path.join(MODEL_DIR, f'{MODEL_NAME}.json')):
        os.remove(mp)

    if os.path.exists(RESULT_DIR):
        for f in os.listdir(RESULT_DIR):
            os.remove(os.path.join(RESULT_DIR, f))
        os.rmdir(RESULT_DIR)

def init():
    download_1M_5m(mega_ticks)

    clean_model_and_test_results()
    train_model(MODEL_NAME)
    test_model(MODEL_NAME)
