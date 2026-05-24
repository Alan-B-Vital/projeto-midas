import mplfinance as mpf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def fplot(df: pd.DataFrame):
    tcdf = df[['Lower','Upper','SMA']]
    apd = mpf.make_addplot(tcdf)
    mpf.plot(df, figratio=(8,4), type='candle', addplot=apd, volume=False, style='yahoo')

def plot(y1, y2):
    ## values
    plt.figure(figsize=(12,6))
    plt.plot(y1.values, label='Real')
    plt.plot(y2, label='Predito')
    plt.legend()
    plt.title('Real vs Predito')
    plt.xlabel('Tempo')
    plt.ylabel('Preço')

    ## error rate
    plt.figure(figsize=(12,6))
    error = np.abs(y1 - y2)
    plt.plot(error, label='Error')

    plt.show()