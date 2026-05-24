import numpy as np
import pandas as pd
import os

TEST_SAMPLE_DIR = './samples/test'
RESULT_DIR = './results'
CONFIG = {
    'buy_threshold': 0.85,
    'sell_threshold': 0.85,
    'cost': 0.0005,
    'annualization': 252 * 78
}

def generate_signals(pred_proba, buy_th, sell_th):
    pred = np.argmax(pred_proba, axis=1)
    confidence = np.max(pred_proba, axis=1)
    signals = np.zeros(len(pred))
    signals[
        (pred == 2) & (confidence > buy_th)
    ] = 1
    signals[
        (pred == 0) & (confidence > sell_th)
    ] = -1

    # anti-lookahead
    signals = np.roll(signals, 1)
    signals[0] = 0

    return signals

def simulate_strategy(
    future_returns,
    signals,
    cost
):
    strategy_returns = signals * future_returns
    trades = np.diff(signals, prepend=0) != 0
    strategy_returns -= trades * cost
    equity = np.cumprod(1 + strategy_returns)

    return {
        'equity': equity,
        'returns': strategy_returns,
        'trades': trades
    }

def calculate_metrics(
    equity,
    strategy_returns,
    trades,
    signals,
    annualization
):
    total_return = equity[-1] - 1
    trade_count = np.sum(trades)
    active_returns = strategy_returns[
        signals != 0
    ]

    win_rate = 0
    avg_trade = 0

    if len(active_returns) > 0:
        win_rate = np.mean(active_returns > 0)
        avg_trade = np.mean(active_returns)

    sharpe = 0

    if np.std(strategy_returns) > 0:
        sharpe = (
            np.mean(strategy_returns)
            / np.std(strategy_returns)
        ) * np.sqrt(annualization)

    max_drawdown = np.min(
        equity / np.maximum.accumulate(equity) - 1
    )

    return {
        'return': total_return,
        'trades': trade_count,
        'win_rate': win_rate,
        'avg_trade': avg_trade,
        'sharpe': sharpe,
        'max_drawdown': max_drawdown,
        'final_equity': equity[-1]
    }

def benchmark(csv_sample):
    print(f"\n{csv_sample}")

    full_path = os.path.join(
        RESULT_DIR,
        f'{csv_sample}.csv'
    )
    df = pd.read_csv(full_path)
    future_returns = df['future_return'].values
    pred_proba = df[
        ['pSELL', 'pHOLD', 'pBUY', 'confidence']
    ].values
    signals = generate_signals(
        pred_proba,
        CONFIG['buy_threshold'],
        CONFIG['sell_threshold']
    )
    
    confidance  = pred_proba[:, 3]

    sim = simulate_strategy(
        future_returns,
        signals,
        CONFIG['cost']
    )

    metrics = calculate_metrics(
        sim['equity'],
        sim['returns'],
        sim['trades'],
        signals,
        CONFIG['annualization']
    )

    print(
        f"return={metrics['return']:.2%}, "
        f"trades={metrics['trades']}, "
        f"win_rate={metrics['win_rate']:.2%}, "
        f"avg_trade={metrics['avg_trade']:.4%}, "
        f"sharpe={metrics['sharpe']:.2f}, "
        f"max_dd={metrics['max_drawdown']:.2%}, "
        f"final_equity={metrics['final_equity']:.4f}"
    )