# -*- coding:utf-8 -*-

import pandas as pd
from database import engine_quant
import datetime as dt
import numpy as np
from sklearn import linear_model
from matplotlib import pyplot as plt
import math

data = pd.read_sql('select from_unixtime(date) as date, close from kline_1d where pair = "BTC-USDT" order by date', engine_quant())
date_range = pd.date_range(data.iloc[0]['date'], data.iloc[-1]['date'])
data.index = data['date']
data = data.reindex(index=date_range, method='ffill')
data['R'] = data['close'] / data['close'].shift(1)
data.dropna(inplace=True)
del data['date']
data['R'] = np.log(data['R'])

def hurst(ls):
    #计算对数序列
    # data['R'] = np.log(data['R'])
    M = len(ls) #数据长度
    #按长度A均分序列为n份
    RSn = []
    ERS = []
    Nx = []
    Vnn = []

    for A in range(2, 1+int((M-1)/2)):
        n = int(M / A)
        N = n * A #保证均分的情况下，部分数据会被剔除，数量为M - N
        RS = []
        Vn = []

        for i in range(M-N, M-1, A):
            Ia = ls[i:i+A]
            Ea = np.mean(Ia) #每一分组中的均值
            Xka = []

            for k in range(A):
                Xka.append(Ia[0:k+1].sum() - Ea * (k+1))
            Ra = max(Xka) - min(Xka)
            Sa = Ia.std()
            if Sa == 0:
                Sa = 0.0001

            RS.append(Ra / Sa)
            Vn.append(Ra / np.sqrt(A))

        RSn.append(sum(RS) / len(RS))
        Nx.append(A)
        Vnn.append(sum(Vn) / len(Vn))
        ES = ((A - 0.5) / A)* math.pow(A * np.pi / 2, -0.5) * sum([math.sqrt((A - r) / r) for r in range(1, A)])
        ERS.append(ES)

    # print(Nx, RSn)
    regr = linear_model.LinearRegression()
    regr.fit(np.log(Nx).reshape(-1,1), np.log(RSn).reshape(-1,1))
    H = regr.coef_[0][0]
    regr.fit(np.log(Nx).reshape(-1,1), np.log(ERS).reshape(-1,1))
    EH = regr.coef_[0][0]

    # print(H)
    # plt.plot(np.log(Nx), np.log(RSn))
    # plt.show()
    # plt.plot(np.log(Nx), Vnn)
    # plt.show()
    return H

def cal_moment(ls):
    regr = linear_model.LinearRegression()
    x = np.array(list(range(len(ls))))
    y = np.array(ls)
    regr.fit(x.reshape(-1, 1), y.reshape(-1, 1))
    # print(regr.coef_[0][0])
    return regr.coef_[0][0]

def trade(data, cash=1000000, window = 120, cost=0.001, date_range=['2018-01-01', (dt.date.today() - dt.timedelta(1)).strftime('%Y-%m-%d')]):
    # data['signal_moment'] = 0
    # data['signal_hurst'] = 0
    df = data.copy()
    df['signal_moment'] = df['R'].rolling(window).apply(cal_moment, raw=False)
    df['signal_hurst'] = df['R'].rolling(window).apply(hurst, raw=False)
    df['signal_buy'] = (((df['signal_hurst'] > 0.6) & (df['signal_hurst'].shift(1) < 0.6)) & (df['signal_moment'] > 0)) | (((df['signal_hurst'] < 0.4) & (df['signal_hurst'].shift(1) > 0.4)) & (df['signal_moment'] < 0))
    df['signal_sell'] = (((df['signal_hurst'] > 0.6) & (df['signal_hurst'].shift(1) < 0.6)) & (df['signal_moment'] < 0)) | (((df['signal_hurst'] < 0.4) & (df['signal_hurst'].shift(1) > 0.4)) & (df['signal_moment'] > 0))

    # for i in range(window, len(data)):
    #     ls = np.array(df.iloc[i-window:i]['R'].tolist())
    #     df.loc[df.index[i], 'signal_moment'] = cal_moment(ls)
    #     df.loc[df.index[i], 'signal_hurst'] = hurst(df.iloc[i-window:i])

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(df.index, df.signal_hurst)
    ax2 = ax1.twinx()
    ax2.plot(df.index, df.close, color='r')
    plt.show()

    date_list = list(
        pd.date_range(dt.datetime.strptime(date_range[0], '%Y-%m-%d'), dt.datetime.strptime(date_range[-1], '%Y-%m-%d'), freq='7D'))
    coin_id = 'BTC-USDT'
    coin_pool = {coin_id:[0.0,0.0,0.0]}
    price = []

    for date in date_list:
        df_day = df.loc[date]
        if df_day['signal_buy'] and cash > 0:
            coin_pool[coin_id][1] = coin_pool[coin_id][1] + cash
            coin_pool[coin_id][0] = coin_pool[coin_id][0] + cash / (1 + cost) / df_day['close']
            cash = 0
        elif df_day['signal_sell'] and coin_pool[coin_id][0] > 0:
            cash = cash + coin_pool[coin_id][0] * df_day['close'] * (1 - cost)
            coin_pool[coin_id][2] += coin_pool[coin_id][0] * df_day['close'] * (1 - cost) - \
                                     coin_pool[coin_id][1]
            coin_pool[coin_id][0] = 0
            coin_pool[coin_id][1] = 0
        marketcap = coin_pool[coin_id][0] * df_day['close']
        total = cash + marketcap
        price.append(total / 1000000.)
        print(price[-1])
    plt.plot(date_list, price)
    plt.show()

    return price[-1]

if __name__ == '__main__':
    # df = pd.DataFrame(np.random.rand(2000),columns=['R'])
    # H = []
    # for i in range(60, len(data)):
    #     H.append(hurst(data.iloc[0:i]))
    #     print(H[-1])
    # A = list(range(60,len(data)))
    # plt.plot(A, H)
    # plt.show()
    # plt.hist(data['R'])
        # print(hurst(data))
    trade(data)
    # hurst(data)
