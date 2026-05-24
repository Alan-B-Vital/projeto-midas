
import pandas as pd
import numpy as np
import os

from multiprocessing import Pool, cpu_count
from xgboost import XGBClassifier
from download_samples import assert_dir
from backtest import TEST_SAMPLE_DIR
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_sample_weight

TRAIN_TEST_SPLIT = 0.8
MODEL_DIR = './models'
RESULT_DIR = './results'
TRAIN_SAMPLE_DIR = './samples/train'
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


def split_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    split = int(len(df) * TRAIN_TEST_SPLIT)
    X = df[STATISTICS]
    Y = df[TARGET]

    x_train = X[:split].copy()
    x_test = X[split:].copy()

    y_train = Y[:split].copy()
    y_test = Y[split:].copy()

    return (x_train, x_test, y_train, y_test)


def dataframe_add_statistics(df: pd.DataFrame) -> pd.DataFrame:

    close = df['Close']

    # =========================
    # Moving averages
    # =========================

    sma_10 = close.rolling(10).mean()
    sma_20 = close.rolling(20).mean()

    ema_10 = close.ewm(span=10).mean()
    ema_20 = close.ewm(span=20).mean()

    # distância percentual do preço para médias
    df['dist_sma_10'] = (close - sma_10) / sma_10
    df['dist_sma_20'] = (close - sma_20) / sma_20

    df['dist_ema_10'] = (close - ema_10) / ema_10
    df['dist_ema_20'] = (close - ema_20) / ema_20

    # =========================
    # Volatilidade
    # =========================

    std_10 = close.rolling(10).std()
    std_20 = close.rolling(20).std()

    # volatilidade relativa
    df['volatility_10'] = std_10 / sma_10
    df['volatility_20'] = std_20 / sma_20

    # Bollinger position
    upper_20 = sma_20 + 2 * std_20
    lower_20 = sma_20 - 2 * std_20

    df['bb_position'] = (close - lower_20) / (upper_20 - lower_20)

    # =========================
    # Momentum
    # =========================

    df['return_1'] = close.pct_change(1)
    df['return_3'] = close.pct_change(3)
    df['return_5'] = close.pct_change(5)
    df['return_10'] = close.pct_change(10)

    # =========================
    # Candle structure
    # =========================

    df['high_low_range'] = (df.High - df.Low) / close

    df['open_close_range'] = (df.Close - df.Open) / df.Open

    # =========================
    # Lagged returns
    # =========================

    df['lag_return_1'] = df['return_1'].shift(1)
    df['lag_return_2'] = df['return_1'].shift(2)
    df['lag_return_3'] = df['return_1'].shift(3)

    # =========================
    # Volume
    # =========================

    if 'Volume' in df.columns:

        vol_ma_10 = df.Volume.rolling(10).mean()

        df['relative_volume'] = df.Volume / vol_ma_10

    # =========================
    # Target
    # =========================
    TARGET_MOVE = 0.001
    TARGET_SHIFT = 12

    df['future_return'] = df.Close.shift(-TARGET_SHIFT) / df.Close - 1

    df['target'] = 0

    df.loc[df.future_return > TARGET_MOVE, 'target'] = 1
    df.loc[df.future_return < -TARGET_MOVE, 'target'] = -1

    df['target_ml'] = df['target'].map(CLASS_MAP)
    return df.dropna()

def fetch_model(model_path) -> XGBClassifier:
    assert_dir(MODEL_DIR)
    
    if os.path.exists(
        model_path
    ):
        # print(f'Loading model: {model_path} - ', end='')
        model = XGBClassifier()
        model.load_model(model_path)
        return model

    raise LookupError(f'404: Model in {model_path} not there.')

def save_results(y_test, future_returns, pred_proba, csv_sample):
    assert_dir(RESULT_DIR)
    pred_class = np.argmax(
        pred_proba,
        axis=1
    )
    acc = np.mean(
        pred_class == y_test.values
    )
    error_rate = 1 - acc
    report = classification_report(
        y_test,
        pred_class,
        target_names=[
            'SELL',
            'HOLD',
            'BUY'
        ]
    )

    
    print(
        f"{csv_sample} | "
        f"Accuracy: {acc*100:.2f}% | "
        f"Error: {error_rate*100:.2f}%\n",
        report
    )

    with open(f'{RESULT_DIR}/{csv_sample}.csv', 'w+') as res:
        res.write('future_return,pSELL,pHOLD,pBUY,target,predicted,confidence\n')
        for (target, future_ret, probs, pred) in zip(y_test, future_returns, pred_proba, pred_class):
            confidence = np.max(probs)
            res.write(
                f'{future_ret:.6f},'
                f'{probs[0]:.6f},'
                f'{probs[1]:.6f},'
                f'{probs[2]:.6f},'
                f'{target},'
                f'{pred},'
                f'{confidence}\n',
            )

def train_model(strategy: str) -> None:
    assert_dir(MODEL_DIR)
    dfs = []
    for s in [sample.split('.')[0] for sample in os.listdir(TRAIN_SAMPLE_DIR)]:
        full_path = os.path.join(TRAIN_SAMPLE_DIR, f'{s}.csv')
        _df = pd.read_csv(full_path, parse_dates=True, index_col=0)
        dfs.append(dataframe_add_statistics(_df))

    df = pd.concat(dfs)
    
    x_train = df[STATISTICS].copy()
    y_train = df[TARGET].copy()
    model_path = os.path.join(MODEL_DIR, f'{strategy}.json')

    if os.path.exists(model_path):
        bst = fetch_model(model_path)
    else:
        bst = XGBClassifier(
            n_estimators=600,
            max_depth=5,
            learning_rate=0.06,
            objective='multi:softprob',
            num_class=3
        )
    sample_weights = compute_sample_weight(
        class_weight='balanced',
        y=y_train
    )
    # fit model
    bst.fit(x_train, y_train, sample_weight=sample_weights)
    bst.save_model(model_path)


def predict(df: pd.DataFrame, model_name) -> tuple[pd.Series, any]:
    x_test = df[STATISTICS].copy()
    y_test = df[TARGET].copy()
    model_path = os.path.join(MODEL_DIR, f'{model_name}.json')

    bst = fetch_model(model_path)
    y_pred_proba = bst.predict_proba(x_test)

    return y_test, y_pred_proba

def _test(kwargs):
    full_path = os.path.join(TEST_SAMPLE_DIR, f'{kwargs["csv_sample"]}.csv')
    df = pd.read_csv(full_path, parse_dates=True, index_col=0)
    df = dataframe_add_statistics(df)

    y_test, y_pred_proba = predict(df, kwargs["model_name"])
    save_results(y_test, df['future_return'], y_pred_proba, kwargs["csv_sample"])

def test_model(model_name: str) -> None:
    pool = Pool(processes=(cpu_count() - 1))
    samples = os.listdir(TEST_SAMPLE_DIR)
    test_data = [{"csv_sample": sample.split('.')[0], "model_name": model_name} for sample in samples]

    pool.map(_test, test_data)

    pool.close()
    pool.join()
