import pandas as pd
from database import engine_test
import datetime as dt
from matplotlib import pyplot as plt
import math
import numpy as np


def RSI(t,df):
    df['increase'] = df['close_p'] / df['close_p'].shift(1) - 1
    df['dex'] = np.nan

    for i in range(t-1,len(df)):
        df_window = df.iloc[i-t+1:i+1]
        df_up = df_window[df_window['increase'] > 0]
        #df_tie = df_window[df_window['increase'] == 0]
        df_down = df_window[df_window['increase'] < 0]

        up = df_up['increase'].mean()
        down = df_down['increase'].mean()
        rsi = round(up / (up - down) * 100, 2)
        df.iloc[i, 5] = rsi

    return df

def WR(t,df):
    df['dex'] = np.nan

    for i in range(t-1,len(df)):
        df_window = df.iloc[i-t+1:i+1]
        price_max = df_window['close_p'].max()
        price_min = df_window['close_p'].min()
        price_now = df_window.iloc[-1]['close_p']
        wr = round((price_max - price_now) / (price_max - price_min) * 100, 2)
        df.iloc[i, 4] = wr

    return df

def VR(t,df):
    df['increase'] = df['close_p'] / df['close_p'].shift(1) - 1
    df['dex'] = np.nan

    for i in range(t-1,len(df)):
        df_window = df.iloc[i-t+1:i+1]
        df_up = df_window[df_window['increase'] > 0]
        df_tie = df_window[df_window['increase'] == 0]
        df_down = df_window[df_window['increase'] < 0]
        vr = round((df_up['volume'].sum() + 0.5 * df_tie['volume'].sum()) / (
                    df_down['volume'].sum() + 0.5 * df_tie['volume'].sum()) * 100, 2)
        df.iloc[i, 5] = vr

    return df


if __name__ == '__main__':
    '''
    测试结果：取10-12之间的数较为合适，选择12
    '''
    coin_id = 1341
    sql = "select day,coin_id,close_p,volume from qy_coin_data where coin_id = {} and close_p > 0 order by day".\
        format(coin_id)
    df = pd.read_sql(sql, engine_test())
    date_list = list(pd.date_range(df.iloc[0]['day'], df.iloc[-1]['day']))
    df = df.copy()
    df.index = pd.to_datetime(df['day'])
    df = df.reindex(index=date_list,method='ffill')

    store = [0,0]

    for t in range(7, 21):
        df = VR(t, df)
        cash = 1000000
        cost = 0.002
        acount = cash
        stock = 0
        wealth = cash

        for date in date_list:
            df_coin = df.loc[date]
            if df_coin['dex'] < 40 and cash > 0:
                stock = cash / df_coin['close_p']
                cash = 0
            elif df_coin['dex'] > 150 and stock > 0:
                cash += stock * df_coin['close_p'] * (1-cost)
                stock = 0

            wealth = cash + stock * df_coin['close_p']

        print(t, wealth)

        # if wealth > store[0]:
        #     store[0] = wealth
        #     store[1] = t
        #     print(store)