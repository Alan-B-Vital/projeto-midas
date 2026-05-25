TEST_SAMPLE_DIR = './samples/test'
TRAIN_SAMPLE_DIR = './samples/train'
RESULT_DIR = './results'
MODEL_DIR = './models'
DUMP_DIR = './dumps'

MODEL_NAME = 'mega'
TRADE_CONFIG = {
    'buy_threshold': 0.60,
    'sell_threshold': 0.60,
    'cost': 0.0005,
    'annualization': 252 * 78,
    'max_window': 500
}
TARGET_SHIFT = 12
TRAIN_TEST_SPLIT = 0.8
STATISTICS = [
    'dist_sma_10',
    'dist_sma_20',
    'dist_ema_10',
    'dist_ema_20',
    'volatility_10',
    'volatility_20',
    'bb_position',
    'return_1',
    'return_3',
    'return_5',
    'return_10',
    'high_low_range',
    'open_close_range',
    'lag_return_1',
    'lag_return_2',
    'lag_return_3',
    'relative_volume'
]
TARGET = 'target_ml'


CLASS_MAP = {
    -1: 0,  # SELL
     0: 1,  # HOLD
     1: 2   # BUY
}

INV_CLASS_MAP = {
    0: -1,
    1: 0,
    2: 1
}
