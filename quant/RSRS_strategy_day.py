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

def data(coin_id):
    sql = '''select date,pair,low,high,close,volume from kline_1d where pair = "{}" and from_unixtime(date) >= "2017-11-01" order by date'''.format(
        coin_id)
    df_coin = pd.read_sql(sql, engine)
    df_coin['date'] = df_coin['date'].apply(
        lambda x: dt.datetime(time.localtime(int(x)).tm_year, time.localtime(int(x)).tm_mon,
                              time.localtime(int(x)).tm_mday))
    regr = linear_model.LinearRegression()
    # df_new = pd.DataFrame()
    date_list = pd.date_range(df_coin.date[0], df_coin.iloc[-1]['date'], freq='d')
    df_coin = df_coin.copy()
    df_coin.index = df_coin['date']
    df_coin.sort_index(ascending=True, inplace=True)
    df_coin = df_coin.reindex(index=date_list, method='ffill')
    df_coin.fillna(0, inplace=True)
    df_coin.pair = coin_id
    return df_coin

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

def RSRS(N,M,up,down,coin_id):
    sql = '''select date,pair,low,high,close from kline_1d where pair = "{}" and from_unixtime(date) >= "2017-11-01" order by date'''.format(coin_id)
    df_coin = pd.read_sql(sql, engine)
    df_coin['date'] = df_coin['date'].apply(lambda x: dt.datetime(time.localtime(int(x)).tm_year, time.localtime(int(x)).tm_mon, time.localtime(int(x)).tm_mday))
    regr = linear_model.LinearRegression()
    #df_new = pd.DataFrame()
    date_list = pd.date_range(df_coin.date[0], df_coin.iloc[-1]['date'],freq='d')
    if len(df_coin) < M:
        print("don't have enough data")
        return False
    df_coin = df_coin.copy()
    df_coin.index = df_coin['date']
    df_coin.sort_index(ascending=True, inplace=True)
    df_coin = df_coin.reindex(index=date_list, method='ffill')
    df_coin.fillna(0, inplace=True)
    df_coin.pair = coin_id
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

    df_coin['coef_N'] = coef_N
    df_coin['avg'] = df_coin['coef_N'].rolling(M).mean()
    df_coin['std'] = df_coin['coef_N'].rolling(M).std()
    df_coin['RSRS'] = (df_coin['coef_N'] - df_coin['avg']) / df_coin['std']
    #df_coin['RSRS_R2'] = df_coin['RSRS']
    df_coin['gold'] = (df_coin['RSRS'] > up)
    df_coin['dead'] = (df_coin['RSRS'] < down)
    df_coin = df_coin[['date','pair','close','gold', 'dead']]
    #df_coin.to_csv('d://RSRS_bitcoin.csv')

    return df_coin

def MACD(n1,n2,n3,v,coin_id):
    #sql = "select coin_id from qy_coin_info_extend where publish < '2015-09-01' and order by coin_id"
    sql = "select day,coin_id,close_p,volume from qy_coin_data where coin_id = {} and close_p > 0 and day >= '2018-01-01' order by day".\
        format(coin_id)
    df_coin = pd.read_sql(sql, engine_test())

    a1 = 2./ (n1 + 1)
    a2 = 2. / (n2 + 1)
    a3 = 2. / (n3 + 1)

    date_list = list(pd.date_range(df_coin.iloc[0]['day'],df_coin.iloc[-1]['day']))
    df_new = pd.DataFrame()
    coin_pool = {}


    df_coin = df_coin.copy()

    df_coin.index = pd.to_datetime(df_coin['day'])
    df_coin = df_coin.reindex(index=date_list,method='ffill')
    EMA12 = []
    EMA26 = []
    DEA = []
#MACD信号
    for i in range(len(df_coin)):
        if i == 0:
            EMA12.append(df_coin.iloc[0]['close_p'])
            EMA26.append(df_coin.iloc[0]['close_p'])
            DIF = EMA12[-1] - EMA26[-1]
            DEA.append(DIF)
        else:
            EMA12.append(a1 * df_coin.iloc[i]['close_p'] + (1 - a1) * EMA12[-1])
            EMA26.append(a2 * df_coin.iloc[i]['close_p'] + (1 - a2) * EMA26[-1])
            DIF = EMA12[-1] - EMA26[-1]
            DEA.append(a3 * DIF + (1 - a3) * DEA[-1])

    df_coin['EMA12'] = EMA12
    df_coin['EMA26'] = EMA26
    df_coin['DIF'] = df_coin['EMA12'] - df_coin['EMA26']
    df_coin['DEA'] = DEA

    df_coin['gold'] = (df_coin['DEA'].shift(1) >= df_coin['DIF'].shift(1)) & (df_coin['DEA'] < df_coin['DIF'])
    df_coin['dead'] = (df_coin['DEA'].shift(1) < df_coin['DIF'].shift(1)) & (df_coin['DEA'] >= df_coin['DIF'])
#交易量与近5天交易量比较
    #df_coin['vol5'] = (df_coin['volume']).rolling(1).mean()
    df_coin['vol'] = (df_coin['volume']) / df_coin['volume'].shift(1)

    # df_coin['change'] = df_coin['close_p'] / df_coin['close_p'].shift(1) - 1
    # df_coin['rise'] = pd.DataFrame({'rise':df_coin['change'],'zeros':0}).max(1)
    # df_coin['rsi'] = df_coin['rise'].rolling(6).mean() / abs(df_coin['change'].rolling(6).mean()) * 100
    # df_coin['rsi_gold'] = (df_coin['rsi'].shift(1) <= 20) & (df_coin['rsi'] > 20)
    # df_coin['rsi_dead'] = (df_coin['rsi'].shift(1) >= 80) & (df_coin['rsi'] < 80)


    # df_new = pd.concat([df_new,df_coin],join='outer')
    #
    # df_new['day'] = df_new.index
    # df_new.index = df_new['coin_id']
    # del df_new['coin_id']

#初始化资产

    origin = 1000000
    cash = origin
    cost = 0.002
    price = [1]
    tag = 0
    index = 0
    coin_pool.update({coin_id: [0, 0, 0]})

    for date in date_list[1:]:
        df_day = df_coin.loc[date]
        df_day_before = df_coin.loc[date - dt.timedelta(1)]

        backtrace = 1 - price[-1]/max(price[index:])
#最大回撤超过10%，卖出平仓
        # if backtrace > 0.2:
        #     if coin_pool[coin_id][0] > 0:
        #         cash += coin_pool[coin_id][0] * df_day_before['close_p'] * (1 - cost)
        #         coin_pool[coin_id][2] += coin_pool[coin_id][0] * df_day_before['close_p'] * (1 - cost) - coin_pool[coin_id][1]
        #         coin_pool[coin_id][0] = 0
        #         coin_pool[coin_id][1] = 0
        #
        #     index = tag

        #止损
        # if coin_pool[coin_id][0] * df_day_before['close_p']  < (1- 0.1) * coin_pool[coin_id][1]:
        #     cash += coin_pool[coin_id][0] * df_day_before['close_p'] * (1 - cost)
        #     coin_pool[coin_id][2] += coin_pool[coin_id][0] * df_day_before['close_p'] * (1 - cost) - \
        #                              coin_pool[coin_id][1]
        #     coin_pool[coin_id][0] = 0
        #     coin_pool[coin_id][1] = 0

        if df_day['gold'] and cash > 0 and df_day['close_p'] > 0:
            coin_pool[coin_id][1] = coin_pool[coin_id][1] + cash
            coin_pool[coin_id][0] = coin_pool[coin_id][0] + cash / df_day['close_p']
            cash = 0
        elif df_day['dead'] and coin_pool[coin_id][0] > 0 and df_day['close_p'] > 0 and df_day['vol'] > v:
            cash = cash + coin_pool[coin_id][0] * df_day['close_p'] * (1 - cost)
            coin_pool[coin_id][2] += coin_pool[coin_id][0] * df_day_before['close_p'] * (1 - cost) - \
                                     coin_pool[coin_id][1]
            coin_pool[coin_id][0] = 0
            coin_pool[coin_id][1] = 0

        marketcap = coin_pool[coin_id][0] * df_day['close_p']
            #if math.isnan(marketcap):print(coin_id,df_day.loc[coin_id]['close_p'],coin_pool[coin_id])
        total = cash + marketcap

        price.append(total / 1000000.)
        #tag += 1

    price_old = df_coin['close_p'] / df_coin.iloc[0]['close_p']

    #print(price)
    #收益曲线
    plt.plot(price)
    plt.plot(price_old.tolist(), color = 'red')
    plt.show()
    #print(df_coin[['gold','dead']])
    return price[-1]

#交易——————————————————————————————————————————————————————————————
def trade(df_new, cash=1000000, cost=0.001, date_range=['2018-02-12', (dt.date.today() - dt.timedelta(1)).strftime('%Y-%m-%d')]):
    coin_id = df_new.iloc[0]['pair']
    price = [1]
    date_list = list(
        pd.date_range(dt.datetime.strptime(date_range[0], '%Y-%m-%d'), dt.datetime.strptime(date_range[-1], '%Y-%m-%d'), freq='d'))
    coin_pool = {coin_id:[0.0,0.0,0.0]}
    tag = 0
    index = 0
    num = 0
    signal_zhisun = []
    signal_buy = []
    signal_sell = []
    price_buy = []
    price_sell = []
    profit = []

    for date in date_list[1:]:
        df_day = df_new.loc[date]
        df_day_before = df_new.loc[date - dt.timedelta(1)]

        backtrace = 1 - price[-1] / max(price[index:])
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
        #for coin_id in coin_pool:
        # if coin_pool[coin_id][0] * df_day_before['close'] < (1 - 0.1) * coin_pool[coin_id][1]:
        #     cash += coin_pool[coin_id][0] * df_day_before['close'] * (1 - cost)
        #     coin_pool[coin_id][2] += coin_pool[coin_id][0] * df_day_before['close'] * (1 - cost) - \
        #                              coin_pool[coin_id][1]
        #     coin_pool[coin_id][0] = 0
        #     coin_pool[coin_id][1] = 0
        #     signal_zhisun.append(date.strftime('%Y-%m-%d'))
        #if df_day['gold'] and cash >= 10000:
        #     coin_pool[coin_id][1] = coin_pool[coin_id][1] + max(cash / 10, 10000)
        #     coin_pool[coin_id][0] = coin_pool[coin_id][0] + max(cash / 10, 10000) / df_day['close_p']
        #     cash = cash - max(cash / 10, 10000)
        # elif df_day['dead'] and coin_pool[coin_id][0] > 0:
        #     cash = cash + coin_pool[coin_id][0] * df_day['close_p'] * (1 - cost)
        #     coin_pool[coin_id][2] += coin_pool[coin_id][0] * df_day_before['close_p'] * (1 - cost) - \
        #                              coin_pool[coin_id][1]
        #     coin_pool[coin_id][0] = 0
        #     coin_pool[coin_id][1] = 0
        if (df_day['gold']) and coin_pool[coin_id][0] == 0:
            coin_pool[coin_id][1] = coin_pool[coin_id][1] + cash
            coin_pool[coin_id][0] = coin_pool[coin_id][0] + cash / (1 + cost) / df_day['close']
            cash = 0
            signal_buy.append(date)
        elif (df_day['dead']) and coin_pool[coin_id][0] > 0:
            cash = cash + coin_pool[coin_id][0] * df_day['close'] * (1 - cost)
            coin_pool[coin_id][2] += coin_pool[coin_id][0] * df_day['close'] * (1 - cost) - \
                                     coin_pool[coin_id][1]
            coin_pool[coin_id][0] = 0
            coin_pool[coin_id][1] = 0
            signal_sell.append(date)
            profit.append(coin_pool[coin_id][2])
        marketcap += coin_pool[coin_id][0] * df_day['close']
            # if math.isnan(marketcap):print(coin_id,df_day.loc[coin_id]['close_p'],coin_pool[coin_id])
        total = cash + marketcap

        price.append(total / 1000000.)
        tag += 1

    df_profit = pd.DataFrame(profit, columns=['profit'])
    df_profit['profit_m'] = df_profit['profit'] - df_profit['profit'].shift(1)
    # df_profit.iloc[0]['profit_m'] = 0
    # print('盈利数：%d,亏损数：%d' % (len(df_profit[df_profit['profit_m']>0]), len(df_profit[df_profit['profit_m']<0])))
    bitcoin = pd.read_sql(
        "select * from kline_1d  where from_unixtime(date) between '{}' and '{}' and pair = '{}' order by date".format(date_range[0], date_range[1], coin_id),
        engine)
    btc = (bitcoin['close'] / bitcoin.iloc[0]['close']).tolist()
    # bitcoin_h = bitcoin['high'].tolist()
    # bitcoin_l = bitcoin['low'].tolist()
    #print(price)
    #print(pd.DataFrame(coin_pool))
    #print(coin_pool)
    for da in signal_sell:
        price_sell.append(price[date_list.index(da)])
    for da in signal_buy:
        price_buy.append(price[date_list.index(da)])
    plt.plot(date_list,price, label='RSRS')
    plt.plot(date_list,btc, color='red', label='Standard')
    plt.scatter(signal_buy, price_buy, marker='v',c='purple')
    plt.scatter(signal_sell, price_sell, marker='o', c='green')
    plt.show()
    #
    # # plt.plot(bitcoin_h)
    # # plt.plot(bitcoin_l)
    # # plt.show()
    # # print('zhisun: ',signal_zhisun)
    print('buy: ',signal_buy)
    print('sell: ',signal_sell)
    # saf = pd.DataFrame(price)
    # saf.to_csv('d:/rsrs_xrp.csv')
    return price[-1]

if __name__ == '__main__':
    result = 0
    # n1_list = list(range(7,16))
    # n2_list = list(range(20,41))
    # v_list = list(range(8,13))
    coin_id = 1
#参数调优   [13, 31, 8, 0.8]
    # for n1 in n1_list:
    #     for n2 in n2_list:
    #         for n3 in range(5,n1):
    #             for v in v_list:
    #                 profit = MACD(n1, n2, n3, v / 10., coin_id)
    #                 if profit > result:
    #                     store = [n1,n2,n3,v]
    #                     result = profit
    #                     print(profit)
#    print(store)

    # n1,n2,n3,v,coin_id = 13, 31, 8, 0.8, 1
    # print(MACD(n1, n2, n3, v, coin_id))
    #df = data('ETH-BTC')
    RS = RSRS(18,140,1,0.1,'EOS-USDT')
    #df_new = RSI(12,RS)
    #df_new = PSY(12,6,df)
    #df_new = VR(12,df_new)
    print(trade(RS))
#RSRS信号参数调优，BTC(N=14, M=110),XRP(14,130), ETH(15,110), EOS(19, 130), BNB(18, 120), ADA(12, 100),XLM(17, 100),LTC(16,280)

    coin_ids = ['BTC-USDT', 'ETH-USDT', 'XRP-USDT', 'EOS-USDT', 'XLM-USDT', 'LTC-USDT',
                'ETH-BTC', 'XRP-BTC', 'EOS-BTC', 'XLM-BTC', 'LTC-BTC']
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
