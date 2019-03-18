#-*- coding:utf-8 -*-
import pandas as pd
from database import engine_test
import datetime as dt
from matplotlib import pyplot as plt
from sklearn import linear_model
import math
import numpy as np

engine = engine_test()

def RSRS(N,M):
    sql = '''select id,symbol from qy_coin_info order by marketcap desc limit 10
    '''
    basic = pd.read_sql(sql, engine)['id'].tolist()
    sql = '''select * from qy_coin_data where close_p > 0 and day >= "2017-01-01" '''
    df = pd.read_sql(sql, engine)
    regr = linear_model.LinearRegression()
    df_new = pd.DataFrame()
    date_list = pd.date_range('2017-01-01', '2018-11-30')

    for coin_id in basic:
        df_coin = df[df['coin_id'] == coin_id]
        if len(df_coin) < M: continue
        df_coin = df_coin.copy()
        df_coin.index = pd.to_datetime(df_coin['day'])
        df_coin.sort_values(by='day', ascending=True, inplace=True)
        df_coin = df_coin.reindex(index=date_list, method='ffill')
        df_coin.fillna(0, inplace=True)
        df_coin.coin_id = coin_id
        coef_N = []

        for i in range(len(df_coin)):
            if i < N:
                regr.fit(np.array(df_coin.iloc[0:i+1]['low_p']).reshape(-1,1), np.array(df_coin.iloc[0:i+1]['high_p']).reshape(-1,1))
            else:
                regr.fit(np.array(df_coin.iloc[i-N+1:i + 1]['low_p']).reshape(-1, 1),
                                np.array(df_coin.iloc[i-N+1:i + 1]['high_p']).reshape(-1, 1))
            coef_N.append(regr.coef_[0][0])

        df_coin['coef_N'] = coef_N
        df_coin['avg'] = df_coin['coef_N'].rolling(M).mean()
        df_coin['std'] = df_coin['coef_N'].rolling(M).std()
        df_coin['RSRS'] = (df_coin['coef_N'] - df_coin['avg']) / df_coin['std']
        df_coin['gold'] = (df_coin['RSRS'] > 0.7)
        df_coin['dead'] = (df_coin['RSRS'] < -0.7)
        df_new = pd.concat([df_new, df_coin], join='outer')

    df_new['day'] = df_new.index
    df_new.index = df_new['coin_id']

    return df_new


def MACD(n1,n2,n3,v):
    #sql = "select coin_id from qy_coin_info_extend where publish < '2015-09-01' and order by coin_id"
    sql = '''SELECT a.id FROM qy_coin_info a 
inner JOIN 
(SELECT coin_id,AVG(market_cap) AS cap FROM qy_coin_data WHERE DAY > DATE_SUB(CURDATE(), INTERVAL 3 MONTH) GROUP BY coin_id ORDER BY cap DESC LIMIT 50) b ON a.id = b.coin_id 
WHERE a.cate = 1 ORDER BY a.id '''
    basic = pd.read_sql(sql, engine)['id'].tolist()

    sql = "select day,coin_id,close_p,volume from qy_coin_data where close_p > 0 and day between '2018-01-01' and '2018-10-31' order by coin_id,day"
    df = pd.read_sql(sql, engine)

    a1 = 2./ (n1 + 1)
    a2 = 2. / (n2 + 1)
    a3 = 2. / (n3 + 1)

    date_list = list(pd.date_range(dt.datetime.strptime('2018-01-01','%Y-%m-%d'),dt.datetime.strptime('2018-10-31','%Y-%m-%d')))
    df_new = pd.DataFrame()
    coin_pool = {}


    for coin in basic:
        df_coin = df[df['coin_id'] == coin]
        df_coin = df_coin.copy()

        if len(df_coin) < len(date_list) - 10:
            continue
        coin_pool.update({coin: [0,0,0]})

        df_coin.index = pd.to_datetime(df_coin['day'])
        df_coin = df_coin.reindex(index=date_list,method='ffill')
        df_coin['coin_id'] = coin
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
        #df_coin['vol5'] = (df_coin['volume']).rolling(5).mean()
        df_coin['vol'] = df_coin['volume'] / df_coin['volume'].shift(1)

        # df_coin['change'] = df_coin['close_p'] / df_coin['close_p'].shift(1) - 1
        # df_coin['rise'] = pd.DataFrame({'rise':df_coin['change'],'zeros':0}).max(1)
        # df_coin['rsi'] = df_coin['rise'].rolling(6).mean() / abs(df_coin['change'].rolling(6).mean()) * 100
        # df_coin['rsi_gold'] = (df_coin['rsi'].shift(1) <= 20) & (df_coin['rsi'] > 20)
        # df_coin['rsi_dead'] = (df_coin['rsi'].shift(1) >= 80) & (df_coin['rsi'] < 80)


        df_new = pd.concat([df_new,df_coin],join='outer')

    df_new['day'] = df_new.index
    df_new.index = df_new['coin_id']
    del df_new['coin_id']

#初始化资产

    origin = 1000000
    cash = origin
    cost = 0.002
    price = [1]
    tag = 0
    index = 0
    for date in date_list[1:]:
        df_day = df_new[pd.to_datetime(df_new['day']) == date]
        df_day_before = df_new[pd.to_datetime((df_new['day'])) == date - dt.timedelta(1)]

        backtrace = 1 - price[-1]/max(price[index:])
#最大回撤超过10%，卖出亏损的持仓币种
        if backtrace > 0.1:
            for coin_id in coin_pool:
                if coin_pool[coin_id][0] > 0 and coin_pool[coin_id][0] * df_day_before.loc[coin_id]['close_p'] < coin_pool[coin_id][1] * (1 - 0.1):
                    cash += coin_pool[coin_id][0] * df_day_before.loc[coin_id]['close_p'] * (1 - cost)
                    coin_pool[coin_id][2] += coin_pool[coin_id][0] * df_day_before.loc[coin_id]['close_p'] * (1 - cost) - coin_pool[coin_id][1]
                    coin_pool[coin_id][0] = 0
                    coin_pool[coin_id][1] = 0

            index = tag

        marketcap = 0
        #print(date)
        for coin_id in coin_pool.keys():
            if coin_pool[coin_id][0] * df_day_before.loc[coin_id]['close_p']  < (1- 0.1) * coin_pool[coin_id][1]:
                cash += coin_pool[coin_id][0] * df_day_before.loc[coin_id]['close_p'] * (1 - cost)
                coin_pool[coin_id][2] += coin_pool[coin_id][0] * df_day_before.loc[coin_id]['close_p'] * (1 - cost) - \
                                         coin_pool[coin_id][1]
                coin_pool[coin_id][0] = 0
                coin_pool[coin_id][1] = 0
            if df_day.loc[coin_id]['gold'] and cash >= 10000 and df_day.loc[coin_id]['close_p'] > 0:
                coin_pool[coin_id][1] = coin_pool[coin_id][1] + max(cash/10,10000)
                coin_pool[coin_id][0] = coin_pool[coin_id][0] + max(cash/10,10000) / df_day.loc[coin_id]['close_p']

                #print(coin_id,df_day.loc[coin_id]['close_p'],coin_pool[coin_id])
                cash = cash - max(cash/10,10000)
            elif df_day.loc[coin_id]['dead'] and coin_pool[coin_id][0] > 0 and df_day.loc[coin_id]['close_p'] > 0 \
                    and df_day.loc[coin_id]['vol'] > v:
                cash = cash + coin_pool[coin_id][0] * df_day.loc[coin_id]['close_p'] * (1 - cost)
                coin_pool[coin_id][2] += coin_pool[coin_id][0] * df_day_before.loc[coin_id]['close_p'] * (1 - cost) - \
                                         coin_pool[coin_id][1]
                coin_pool[coin_id][0] = 0
                coin_pool[coin_id][1] = 0
                #print(coin_id, df_day.loc[coin_id]['close_p'],coin_pool[coin_id])

            marketcap += coin_pool[coin_id][0] * df_day.loc[coin_id]['close_p']
            #if math.isnan(marketcap):print(coin_id,df_day.loc[coin_id]['close_p'],coin_pool[coin_id])
        total = cash + marketcap

        price.append(total / 1000000.)
        tag += 1


    bitcoin = pd.read_sql("select * from qy_coin_data where day between '2018-01-01' and '2018-10-31' and coin_id = 1 order by day",engine_test())
    bitcoin = (bitcoin['close_p'] / bitcoin.iloc[0]['close_p']).tolist()
    print(price)
    print(pd.DataFrame(coin_pool))
    print(coin_pool)
    plt.plot(price)
    plt.plot(bitcoin,color='red')
    plt.show()

    return price[-1]
#交易——————————————————————————————————————————————————————————————
def trade(df_new, cash=1000000, cost=0.002, date_range=['2018-01-01', '2018-11-30']):
    price = [1]
    date_list = list(
        pd.date_range(dt.datetime.strptime(date_range[0], '%Y-%m-%d'), dt.datetime.strptime(date_range[-1], '%Y-%m-%d')))
    coin_pool = {}

    for id in list(set(df_new['coin_id'])):
        coin_pool.update({id: [0,0,0]})

    tag = 0
    index = 0

    for date in date_list[1:]:
        df_day = df_new[pd.to_datetime(df_new['day']) == date]
        df_day_before = df_new[pd.to_datetime((df_new['day'])) == date - dt.timedelta(1)]

        backtrace = 1 - price[-1] / max(price[index:])
        # 最大回撤超过10%，卖出亏损的持仓币种
        if backtrace > 0.1:
            for coin_id in coin_pool:
                if coin_pool[coin_id][0] > 0 and coin_pool[coin_id][0] * df_day_before.loc[coin_id]['close_p'] < \
                        coin_pool[coin_id][1] * (1 - 0.1):
                    cash += coin_pool[coin_id][0] * df_day_before.loc[coin_id]['close_p'] * (1 - cost)
                    coin_pool[coin_id][2] += coin_pool[coin_id][0] * df_day_before.loc[coin_id]['close_p'] * (
                                1 - cost) - coin_pool[coin_id][1]
                    coin_pool[coin_id][0] = 0
                    coin_pool[coin_id][1] = 0

            index = tag

        marketcap = 0
        # print(date)
        #根据信号及止损线进行调仓
        for coin_id in coin_pool:
            # if coin_pool[coin_id][0] * df_day_before.loc[coin_id]['close_p'] < (1 - 0.1) * coin_pool[coin_id][1]:
            #     cash += coin_pool[coin_id][0] * df_day_before.loc[coin_id]['close_p'] * (1 - cost)
            #     coin_pool[coin_id][2] += coin_pool[coin_id][0] * df_day_before.loc[coin_id]['close_p'] * (1 - cost) - \
            #                              coin_pool[coin_id][1]
            #     coin_pool[coin_id][0] = 0
            #     coin_pool[coin_id][1] = 0
            if df_day.loc[coin_id]['gold'] and cash >= 10000:
                coin_pool[coin_id][1] = coin_pool[coin_id][1] + max(cash / 10, 10000)
                coin_pool[coin_id][0] = coin_pool[coin_id][0] + max(cash / 10, 10000) / df_day.loc[coin_id]['close_p']
                cash = cash - max(cash / 10, 10000)
            elif df_day.loc[coin_id]['dead'] and coin_pool[coin_id][0] > 0:
                cash = cash + coin_pool[coin_id][0] * df_day.loc[coin_id]['close_p'] * (1 - cost)
                coin_pool[coin_id][2] += coin_pool[coin_id][0] * df_day_before.loc[coin_id]['close_p'] * (1 - cost) - \
                                         coin_pool[coin_id][1]
                coin_pool[coin_id][0] = 0
                coin_pool[coin_id][1] = 0
                # print(coin_id, df_day.loc[coin_id]['close_p'],coin_pool[coin_id])

            marketcap += coin_pool[coin_id][0] * df_day.loc[coin_id]['close_p']
            # if math.isnan(marketcap):print(coin_id,df_day.loc[coin_id]['close_p'],coin_pool[coin_id])
        total = cash + marketcap

        price.append(total / 1000000.)
        tag += 1

    bitcoin = pd.read_sql(
        "select * from qy_coin_data where day between '{}' and '{}' and coin_id = 1 order by day".format(date_range[0], date_range[1]),
        engine_test())
    bitcoin = (bitcoin['close_p'] / bitcoin.iloc[0]['close_p']).tolist()
    #print(price)
    #print(pd.DataFrame(coin_pool))
    #print(coin_pool)
    plt.plot(price)
    plt.plot(bitcoin, color='red')
    plt.show()

    return price[-1]

if __name__ == '__main__':
    n1,n2,n3,v = 13,31,8,0.8
    #print(MACD(n1,n2,n3,v))
    # 参数测试结果取N = 16, M = 120
    #temp = 0
    # for n in range(12,20):
    #     for m in range(100,200,10):
    #         RS = RSRS(N=n, M=m)
    #         if temp < trade(RS):
    #             a, b = n, m
    #             temp = trade(RS)
    #
    # print('(N,M):%d, %d\n' % (a,b))
    RS = RSRS(16,120)
    print(trade(RS))





