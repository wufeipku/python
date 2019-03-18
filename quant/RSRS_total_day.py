#-*- coding:utf-8 -*-
import pandas as pd
from database import engine_test, engine_exchange, engine_local,engine_quant
import datetime as dt
from matplotlib import pyplot as plt
import math
from sklearn import linear_model
import numpy as np
import time
import datetime as dt

engine = engine_quant()
coin_ids = ('BTC-USDT', 'ETH-USDT', 'XRP-USDT', 'EOS-USDT', 'XLM-USDT', 'LTC-USDT')

def data():
    sql = '''select date,pair,low,high,close,volume from kline_1d where pair in {} order by date'''.format(coin_ids)
    df = pd.read_sql(sql, engine)
    df_new = pd.DataFrame()

    for id in coin_ids:
        df_coin = df[df['pair']==id]
        df_coin = df_coin.copy()
        df_coin['date'] = df_coin['date'].apply(
            lambda x: dt.datetime(time.localtime(int(x)).tm_year, time.localtime(int(x)).tm_mon,
                                  time.localtime(int(x)).tm_mday))
        date_list = pd.date_range(df_coin.iloc[0]['date'], df_coin.iloc[-1]['date'], freq='d')
        df_coin.index = df_coin['date']
        df_coin.sort_index(ascending=True, inplace=True)
        df_coin = df_coin.reindex(index=date_list, method='ffill')
        df_coin.fillna(0, inplace=True)
        df_coin.pair = id
        df_new = pd.concat([df_new, df_coin], join='outer')
    return df_new

def RSI(t,df):
    df['increase'] = df['close'] / df['close'].shift(1) - 1
    df['dex'] = np.nan

    for i in range(t-1,len(df)):
        df_window = df.iloc[i-t+1:i+1]
        df_up = df_window[df_window['increase'] > 0]
        #df_tie = df_window[df_window['increase'] == 0]
        df_down = df_window[df_window['increase'] < 0]

        up = df_up['increase'].mean()
        down = df_down['increase'].mean()
        rsi = round(up / (up - down) * 100, 2)
        df.iloc[i, 6] = rsi

    return df

def PSY(t, m, df):
    df['increase'] = df['close'] / df['close'].shift(1) - 1
    df['psy'] = np.nan
    columns = list(df.columns)
    index = columns.index('psy')

    for i in range(t-1,len(df)):
        df_window = df.iloc[i-t+1:i+1]
        df_up = df_window[df_window['increase'] > 0]
        #df_tie = df_window[df_window['increase'] == 0]
        psy = len(df_up) / t
        df.iloc[i, index] = psy

    df['psyma'] = df['psy'].rolling(m).mean()
    df['gold'] = (df['psy'].shift(1) < df['psyma'].shift(1)) & (df['psy'] > df['psyma'])
    df['dead'] = (df['psy'].shift(1) > df['psyma'].shift(1)) & (df['psy'] < df['psyma'])

    return df

def VR(t,df):
    df['increase'] = df['close'] / df['close'].shift(1) - 1
    df['vr'] = np.nan
    columns = list(df.columns)
    index = columns.index('vr')

    for i in range(t-1,len(df)):
        df_window = df.iloc[i-t+1:i+1]
        df_up = df_window[df_window['increase'] > 0]
        df_tie = df_window[df_window['increase'] == 0]
        df_down = df_window[df_window['increase'] < 0]
        vr = round((df_up['volume'].sum() + 0.5 * df_tie['volume'].sum()) / (
                    df_down['volume'].sum() + 0.5 * df_tie['volume'].sum()) * 100, 2)
        df.iloc[i, index] = vr

    return df

def RSRS(N,M, coin_id, df):
    regr = linear_model.LinearRegression()
    df_coin = df[df['pair']==coin_id]
    if len(df_coin) < M:
        print("don't have enough data")
        return False
    coef_N = []
    R2 = []

    for i in range(len(df_coin)):
        if i < N-1:
            #regr.fit(np.array(df_coin.iloc[0:i+1]['low']).reshape(-1,1), np.array(df_coin.iloc[0:i+1]['high']).reshape(-1,1))
            coef_N.append(0)
            R2.append(0)
        else:
            regr.fit(np.array(df_coin.iloc[i-N+1:i + 1]['low']).reshape(-1, 1),
                            np.array(df_coin.iloc[i-N+1:i + 1]['high']).reshape(-1, 1))
            coef_N.append(regr.coef_[0][0])
            R2.append(regr.score(np.array(df_coin.iloc[i-N+1:i + 1]['low']).reshape(-1, 1),
                            np.array(df_coin.iloc[i-N+1:i + 1]['high']).reshape(-1, 1)))
    df_coin = df_coin.copy()
    df_coin['coef_N'] = coef_N
    df_coin['avg'] = df_coin['coef_N'].rolling(M).mean()
    df_coin['std'] = df_coin['coef_N'].rolling(M).std()
    df_coin['RSRS'] = (df_coin['coef_N'] - df_coin['avg']) / df_coin['std']
    #df_coin['RSRS_R2'] = df_coin['RSRS']
    df_coin['gold'] = (df_coin['RSRS'] > 1)
    df_coin['dead'] = (df_coin['RSRS'] < 0.1)
    df_coin = df_coin[['pair','close','gold', 'dead']]
    #df_coin.to_csv('d://RSRS_bitcoin.csv')

    return df_coin

#交易——————————————————————————————————————————————————————————————
def trade(df_new, cash=1000000, cost=0.001, date_range=['2018-09-01', (dt.date.today() - dt.timedelta(1)).strftime('%Y-%m-%d')]):
    price = []
    date_list = list(
        pd.date_range(dt.datetime.strptime(date_range[0], '%Y-%m-%d'), dt.datetime.strptime(date_range[-1], '%Y-%m-%d'), freq='d'))

    coin_pool = {}
    profit = {}

    for id in coin_ids:
        coin_pool.update({id: [0.0,0.0,0.0, 0]})
        profit.update({id: []})
    tag = 0
    index = 0
    num = 0
    signal_zhisun = []
    signal_buy = []
    signal_sell = []
    signal_loss = []
    signal_loss1 = []
    price_buy = []
    price_sell = []
    price_loss = []
    price_loss1 = []

    for date in date_list:
        df_day = df_new.loc[date]
        # df_day_before = df_new.loc[date - dt.timedelta(1)]

        # backtrace = 1 - price[-1] / max(price[index:])
        # 最大回撤超过10%，卖出亏损的持仓币种
        # if backtrace > 0.1:
        #     for coin_id in coin_pool:
        #         if coin_pool[coin_id][0] > 0 and coin_pool[coin_id][0] * df_day_before.loc[coin_id]['close_p'] < \
        #                 coin_pool[coin_id][1] * (1 - 0.1):
        #             cash += coin_pool[coin_id][0] * df_day_before.loc[coin_id]['close_p'] * (1 - cost)
        #             coin_pool[coin_id][2] += coin_pool[coin_id][0] * df_day_before.loc[coin_id]['close_p'] * (
        #                         1 - cost) - coin_pool[coin_id][1]
        #             coin_pool[coin_id][0] = 0
        #             coin_pool[coin_id][1] = 0
        #
        #     index = tag

        marketcap = 0
        # print(date)
        #根据信号及止损线进行调仓
        buy_coin = []
        sell_coin = []
        loss_coin = []

        for coin_id in coin_pool:
            df_coin = df_day[df_day['pair'] == coin_id]
            marketcap += coin_pool[coin_id][0] * df_coin.iloc[0]['close']
            # if coin_pool[coin_id][0] * df_coin.iloc[0]['close'] < coin_pool[coin_id][1] * (1 - 0.1):
            #     loss_coin.append(coin_id)
            #     coin_pool[coin_id][3] = 1
            if df_coin.iloc[0]['gold']:
                buy_coin.append(coin_id)
            elif (coin_pool[coin_id][0] > 0 and (df_coin.iloc[0]['dead'])):
                sell_coin.append(coin_id)

        #总资产---------------------------------------------------------------------
        total = cash + marketcap
        price.append(total / 1000000.)
        if (price[-1] > price[tag] * (1 + 0.5)) or (price[-1] < price[tag] * (1 - 0.2)):
            for coin_id in coin_pool:
                if coin_pool[coin_id][0] > 0:
                    df_coin = df_day[df_day['pair'] == coin_id]
                    cash = cash + coin_pool[coin_id][0] * df_coin.iloc[0]['close'] * (1 - cost)
                    coin_pool[coin_id][2] = coin_pool[coin_id][0] * df_coin.iloc[0]['close'] * (1 - cost) - \
                                            coin_pool[coin_id][1]
                    coin_pool[coin_id][0] = 0
                    coin_pool[coin_id][1] = 0
                    signal_loss1.append(date)
                    profit[coin_id].append(coin_pool[coin_id][2])
            tag = len(price) - 1
        else:
            # for coin_id in loss_coin:
            #     df_coin = df_day[df_day['pair']==coin_id]
            #     cash = cash + coin_pool[coin_id][0] * df_coin.iloc[0]['close'] * (1 - cost)
            #     coin_pool[coin_id][2] = coin_pool[coin_id][0] * df_coin.iloc[0]['close'] * (1 - cost) - \
            #                              coin_pool[coin_id][1]
            #     coin_pool[coin_id][0] = 0
            #     coin_pool[coin_id][1] = 0
            #     signal_loss.append(date)
            #     profit[coin_id].append(coin_pool[coin_id][2])

            for coin_id in sell_coin:
                df_coin = df_day[df_day['pair']==coin_id]
                cash = cash + coin_pool[coin_id][0] * df_coin.iloc[0]['close'] * (1 - cost)
                coin_pool[coin_id][2] = coin_pool[coin_id][0] * df_coin.iloc[0]['close'] * (1 - cost) - \
                                         coin_pool[coin_id][1]
                coin_pool[coin_id][0] = 0
                coin_pool[coin_id][1] = 0
                signal_sell.append(date)
                profit[coin_id].append(coin_pool[coin_id][2])

            temp_cash = cash / len(buy_coin) if len(buy_coin) > 0 else 0
            if temp_cash > 10:
                for coin_id in buy_coin:
                    if coin_pool[coin_id][3] > 0:
                        continue
                    df_coin = df_day[df_day['pair'] == coin_id]
                    coin_pool[coin_id][1] = coin_pool[coin_id][1] + temp_cash
                    coin_pool[coin_id][0] = coin_pool[coin_id][0] + temp_cash / (1 + cost) / df_coin.iloc[0]['close']
                    cash = cash - temp_cash
                    signal_buy.append(date)

        # for coin_id in coin_pool:
        #     coin_pool[coin_id][3] = max(0, coin_pool[coin_id][3] - 1)

        # tag += 1
    for id in profit:
        if len(profit[id]) > 0:
            df_profit = pd.DataFrame(profit[id], columns=['profit'])
            print('%s 盈利数：%d,亏损数：%d' % (id, len(df_profit[df_profit['profit']>0]), len(df_profit[df_profit['profit']<0])))
    bitcoin = pd.read_sql(
        "select * from kline_1d  where from_unixtime(date) between '{}' and '{}' and pair = 'BTC-USDT' order by date".format(date_range[0], date_range[1]),
        engine)
    btc = (bitcoin['close'] / bitcoin.iloc[0]['close']).tolist()
    bitcoin_h = bitcoin['high'].tolist()
    bitcoin_l = bitcoin['low'].tolist()
    #print(price)
    #print(pd.DataFrame(coin_pool))
    #print(coin_pool)
    for da in signal_sell:
        price_sell.append(price[date_list.index(da)])
    for da in signal_buy:
        price_buy.append(price[date_list.index(da)])
    for da in signal_loss:
        price_loss.append(price[date_list.index(da)])
    for da in signal_loss1:
        price_loss1.append(price[date_list.index(da)])
    plt.plot(date_list,price, label='RSRS')
    plt.plot(date_list,btc, color='red', label='Standard')
    plt.scatter(signal_buy, price_buy, marker='v',c='purple')
    plt.scatter(signal_sell, price_sell, marker='o', c='green')
    # plt.scatter(signal_loss,price_loss, marker='x', c='red')
    # plt.scatter(signal_loss1, price_loss1, marker='x', c='b')
    plt.show()
    #
    # # plt.plot(bitcoin_h)
    # # plt.plot(bitcoin_l)
    # # plt.show()
    # # print('zhisun: ',signal_zhisun)
    # print('buy: ',signal_buy)
    # print('sell: ',signal_sell)
    # saf = pd.DataFrame(price)
    # saf.to_csv('d:/rsrs_xrp.csv')
    retracement = []
    for i in range(len(price)):
        retracement.append(1 - price[i] / max(price[0:i+1]))

    Max_retracement = max(retracement)

    return price[-1], Max_retracement

if __name__ == '__main__':
    df = data()
    params = {'BTC-USDT': [13,110], 'ETH-USDT': [16,120], 'XRP-USDT': [14, 130], 'EOS-USDT': [18, 140], 'XLM-USDT': [16, 100], 'LTC-USDT': [19, 290]}
    df_new = pd.DataFrame()

    for id in coin_ids:
        df_coin = RSRS(params[id][0], params[id][1], id, df)
        df_new = pd.concat([df_new, df_coin], join='outer')

    #df_new = RSI(12,RS)
    #df_new = PSY(12,6,df)
    #df_new = VR(12,df_new)
    print(trade(df_new))
#RSRS信号参数调优，BTC(N=14, M=110),XRP(14,130), ETH(15,110), EOS(19, 130), BNB(18, 120), ADA(12, 100),XLM(17, 100),LTC(16,280)

    # coin_ids = ['BTC-USDT', 'ETH-USDT', 'XRP-USDT', 'EOS-USDT', 'XLM-USDT', 'LTC-USDT',
    #             'ETH-BTC', 'XRP-BTC', 'EOS-BTC', 'XLM-BTC', 'LTC-BTC']
    # #[N, M] = {'BTC-USDT':(13, 110), 'ETH-USDT':(16, 120), 'XRP-USDT':(14, 130), 'EOS-USDT':(19, 100), 'XLM-USDT': (16, 100), 'LTC-USDT': (19, 290),
    #             # 'ETH-BTC': (13, 110), 'XRP-BTC': (15, 240), 'EOS-BTC': (19, 110), 'XLM-BTC': (16, 100), 'LTC-BTC':(16, 260)}
    # for coin_id in coin_ids:
    #     temp = 0
    #     for n in range(12,20):
    #         for m in range(100,300,10):
    #             RS = RSRS(N=n, M=m,coin_id=coin_id)
    #
    #             if temp < trade(RS):
    #                 a, b = n, m
    #                 temp = trade(RS)
    #     print('(coin_id,N,M):%s, %d, %d\n' % (coin_id, a, b))
    #     print(temp)
    #     print('-----------------------------------------')
