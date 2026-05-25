import os
import yfinance as yf
import numpy as np
import pandas as pd

from ai import fetch_model, dataframe_add_statistics, MODEL_DIR, STATISTICS, assert_dir
from config import TRADE_CONFIG, DUMP_DIR, TARGET_SHIFT
from datetime import datetime, timedelta

class LiveTrader:
    def __init__(self, model_name: str, symbol: str):
        self.symbol = symbol
        self.model_name = model_name

        model_path = os.path.join(MODEL_DIR, f'{model_name}.json')
        self.model = fetch_model(model_path)

        try:
            dump_path = d_last_week_tick(symbol)
            self.df = pd.read_csv(dump_path, parse_dates=True, index_col=0)
            os.remove(dump_path)
        except:
            self.df = pd.DataFrame()

        self.position = 0
        self.entry_price = None
        self.hold_bars = 0

    def on_new_candle(self, candle: dict):
        """
        candle:
        {
            'Open': ...,
            'High': ...,
            'Low': ...,
            'Close': ...,
            'Volume': ...
        }
        """

        self.df = pd.concat([
            self.df,
            pd.DataFrame([candle])
        ])

        # precisa de candles suficientes
        if len(self.df) < 200:
            return
        
        self.df = self.df.tail(TRADE_CONFIG['max_window'])

        stats_df = dataframe_add_statistics(
            self.df.copy()
        )

        if stats_df.empty:
            return

        latest = stats_df.iloc[-1:]

        x_live = latest[STATISTICS]

        if self.position == 1:
            self.hold_bars -= 1

            if self.hold_bars <= 0:
                self.sell(latest['Close'].iloc[0])

        proba = self.model.predict_proba(x_live)[0]

        self.execute_signal(
            proba,
            latest['Close'].iloc[0]
        )

    def execute_signal(self, buy_prob, sell_prob, close_price):
        if buy_prob >= TRADE_CONFIG['buy_threshold']:
            self.buy(close_price)

        elif sell_prob >= TRADE_CONFIG['sell_threshold']:
            self.sell(close_price)


    def execute_signal(self, proba, close_price):
        pred = np.argmax(proba)
        confidence = np.max(proba)

        # print(
        #     f'pred={pred} '
        #     f'confidence={confidence:.4f} '
        #     f'proba={proba}'
        # )

        # BUY
        if (pred == 2 and confidence >= TRADE_CONFIG['buy_threshold']):
            self.buy(close_price)

        # SELL
        elif (pred == 0 and confidence >= TRADE_CONFIG['sell_threshold']):
            self.sell(close_price)

    def buy(self, price):
        if self.position == 0:
            self.position = 1
            self.hold_bars = TARGET_SHIFT
            self.entry_price = price

            print(f'BUY {self.symbol} @ {price}')

    def sell(self, price):
        if self.position == 1:
            pnl = (price - self.entry_price) / self.entry_price

            print(
                f'SELL {self.symbol} @ {price} '
                f'PnL={pnl:.2%} '
                f'On Hold For={TARGET_SHIFT - self.hold_bars} candels'
            )

            self.position = 0
            self.entry_price = None
            


def d_last_week_tick(tick: str) -> str:
    dates = {
        'start': (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d'),
        'end': datetime.now()
    }
    assert_dir(DUMP_DIR)
    dump_path = os.path.join(DUMP_DIR, f'{tick}_1M_5m.csv')

    df = yf.download(tick, start=dates['start'], end=dates['end'], interval='5m', multi_level_index=False)
    df.to_csv(dump_path)

    return dump_path
