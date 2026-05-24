import numpy as np
import pandas as pd
import os

TEST_SAMPLE_DIR = './samples/test'
RESULT_DIR = './results'
CONFIDENCE_THRESHOLD = {
    'UP': 0.65,
    'DOWN': 0.35
}


def backtest(future_returns, pred_proba, cost=0.0005):

    sell_prob = pred_proba[:, 0]
    hold_prob = pred_proba[:, 1]
    buy_prob  = pred_proba[:, 2]

    signals = np.zeros(len(pred_proba))

    signals[buy_prob > CONFIDENCE_THRESHOLD['DOWN']] = 1
    signals[sell_prob > CONFIDENCE_THRESHOLD['UP']] = -1

    # evita lookahead
    signals = np.roll(signals, 1)
    signals[0] = 0

    # pnl
    strategy_returns = signals * future_returns

    # mudança posição
    trades = signals[1:] != signals[:-1]

    # custos
    strategy_returns[1:] -= trades * cost

    # equity
    equity = np.cumprod(1 + strategy_returns)

    return equity, strategy_returns, signals

def benchmark(csv_sample):

    print(f"\n{csv_sample}:")

    full_path = os.path.join(
        RESULT_DIR,
        f'{csv_sample}.csv'
    )

    df = pd.read_csv(full_path)

    future_returns = df['future_return'].values

    pred_proba = df[
        ['pSELL', 'pHOLD', 'pBUY']
    ].values

    equity, strat_ret, signals = backtest(
        future_returns,
        pred_proba,
        cost=0.0005
    )

    total_return = equity[-1] - 1

    trades = np.sum(
        signals[1:] != signals[:-1]
    )

    active_returns = strat_ret[
        signals != 0
    ]

    if len(active_returns) > 0:

        win_rate = np.mean(
            active_returns > 0
        )

        avg_trade = np.mean(
            active_returns
        )

    else:

        win_rate = 0
        avg_trade = 0

    sharpe = 0

    if np.std(strat_ret) > 0:

        sharpe = (
            np.mean(strat_ret)
            / np.std(strat_ret)
        ) * np.sqrt(252 * 78)

    max_drawdown = np.min(
        equity / np.maximum.accumulate(equity) - 1
    )

    print(
        f"return={total_return:.2%}, "
        f"trades={trades}, "
        f"win_rate={win_rate:.2%}, "
        f"avg_trade={avg_trade:.4%}, "
        f"sharpe={sharpe:.2f}, "
        f"max_dd={max_drawdown:.2%}, "
        f"final_equity={equity[-1]:.4f}"
    )