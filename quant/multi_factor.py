import pandas as pd
from database import engine_test
from sklearn import linear_model
import numpy as np
from matplotlib import pyplot as plt

engine = engine_test()

def m_factor(date_str):
    df = pd.read_sql('select coin_id,day,open_p,close_p,market_cap from qy_coin_data where day > date_sub("{}",interval 3 month) and day <= "{}" order by coin_id,day'\
                     .format(date_str,date_str),engine)
    coin_ids = pd.read_sql('select coin_id from qy_coin_data where day = date_sub("{}",interval 3 month) order by market_cap desc limit 1000'.\
                           format(date_str),engine)['coin_id'].tolist()
    df = df[df.coin_id.isin(coin_ids)]
    df['y'] = df['close_p'] / df['open_p'] - 1
    df.reset_index(drop=1,inplace=True)
    statistic_date = df[df['coin_id']==1]['day'].tolist()
    df_market = pd.DataFrame(columns=['day','market_factor','total_market'] )

    i = 0
    for date in statistic_date:
        df_coin = df[df['day']==date]
        df_coin.index = df_coin['coin_id']
        del df_coin['coin_id']
        df_coin = df_coin.sort_values(by='market_cap',ascending=False)
        # df_coin_big = df_coin.iloc[0:int(len(df_coin)/2)]
        # df_coin_small = df_coin.iloc[int(len(df_coin)/2):]
        #
        # df_coin_big['adjust_y'] = df_coin_big['y'] * df_coin_big['market_cap'] / df_coin_big['market_cap'].sum()
        # df_coin_small['adjust_y'] = df_coin_small['y'] * df_coin_small['market_cap'] / df_coin_small['market_cap'].sum()
        df_coin['adjust_y'] = df_coin['y'] * df_coin['market_cap']
        df_100p = df_coin.iloc[0:int(len(df_coin)/2)].adjust_y.sum() / df_coin.iloc[0:int(len(df_coin)/2)]['market_cap'].sum()
        df_100m = df_coin.iloc[int(len(df_coin)/2):].adjust_y.sum() / df_coin.iloc[int(len(df_coin)/2):]['market_cap'].sum()
        df_all = df_coin.adjust_y.sum() / df_coin['market_cap'].sum()
        df_market.loc[i] = [date,df_100p - df_100m,df_all]
        i += 1

    record = []

    for coin_id in coin_ids:
        regr = linear_model.LinearRegression()
        df_coin = df[df['coin_id']==coin_id]
        df_coin = df_coin.sort_values(by='day',ascending=True)
        df_coin.index = df_coin['day']
        df_coin = df_coin.reindex(statistic_date)
        df_coin = df_coin.copy()
        df_coin['market_factor'] = df_market['market_factor'].tolist()
        df_coin['total_market'] = df_market['total_market'].tolist()
        df_coin = df_coin.dropna(axis=0,how='any')

        if len(df_coin) < 30:
            continue

        df_y = df_coin.loc[:,'y'].values
        y = df_y.T
        df_x = df_coin.drop(['coin_id','close_p','day','market_cap','open_p','y'],1)
        #print(coin_id,df_x.corr())
        x = df_x.values

        #线性回归
        regr.fit(x,y)

        #print(type(regr.coef_.tolist()[0]),regr.intercept_.tolist())
        record.append({'coin_id':coin_id,'market_factor':regr.coef_.tolist()[0],\
                       'total_market':regr.coef_.tolist()[1],'intercept':regr.intercept_,'R2_value':regr.score(x,y)})


    da = pd.DataFrame(record)
    if len(da) == 0:
        return []


    da = da[(da['R2_value'] > 0.4)]
    da = da.sort_values(by='intercept',ascending=True)
    da_new = da.iloc[:30]
    coin_id = da_new['coin_id'].tolist()

    return coin_id

def trade(origin=1000000,cost=0.002):
    wealth = []
    coin_pool = {}
    df = pd.read_sql('select coin_id,day,close_p from qy_coin_data where day >= "2014-01-01" ',engine)
    df['day'] = pd.to_datetime(df['day'])
    statistic_date = pd.date_range('2018-01-01','2018-12-27',freq='QS')
    date_list = pd.date_range('2018-01-01','2018-12-27')
    cash = origin

    for date in date_list:
        df_date = df[df['day']==date]
        #print(df_date)
        temp = 0
        coin_pool1 = coin_pool.copy()
        #计算每日市值，遇到没有数据的情况，直接卖出对应币
        for coin_id in coin_pool1.keys():
            df_coin = df_date[df_date['coin_id']==coin_id]
            try:
                coin_pool[coin_id][1] = coin_pool[coin_id][0] * df_coin.iloc[0]['close_p']
                temp += coin_pool[coin_id][1]
            except:
                cash = cash + coin_pool[coin_id][1] * (1 -cost)
                del coin_pool[coin_id]

        marketcap = temp
        #每个季度第一天根据多因子策略选择成分币，保存在coin_ids列表中
        if date in statistic_date:
            coin_ids = m_factor(date.strftime('%Y-%m-%d'))
            if len(coin_ids) == 0:
                continue

            count = 0

            coin_pool1 = coin_pool.copy()
            #当持仓币不在策略给出的成分币中时，卖出平仓，同时删除持仓记录
            for coin_id in coin_pool1.keys():
                if coin_id not in coin_ids:
                    cash = coin_pool[coin_id][1] * (1 - cost) + cash
                    marketcap =marketcap - coin_pool[coin_id][1]
                    del coin_pool[coin_id]
                else:
                    #记录已经持仓且在策略成分币中的数量
                    count += 1
            #对未持仓的策略成分币的市值进行均分处理
            cash_every = cash / (len(coin_ids) - count)
            #对未持仓的策略成分币进行买入
            for coin_id in coin_ids:
                if coin_id not in coin_pool.keys():
                    df_coin = df_date[df_date['coin_id']==coin_id]
                    try:
                        coin_pool.update({coin_id:[cash_every / df_coin.iloc[0]['close_p'],cash_every]})
                        cash = cash - cash_every
                        marketcap = marketcap + cash_every
                        #temp1 += cash_every
                    except:
                        continue

        print((cash + marketcap)/1000000)

        #print(cash,marketcap)
        wealth.append((cash + marketcap) / 1000000)

    plt.plot(wealth)
    plt.show()








if __name__ == '__main__':
    cash = 1000000
    cost = 0.002
    trade(cash,cost)