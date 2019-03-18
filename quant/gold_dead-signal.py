import pandas as pd
import pymysql
from database import engine_test,engine_online
from matplotlib import pyplot as plt
import datetime as dt
engine = engine_test()

def main():
    sql = "select coin_id from qy_coin_info_extend where publish < '2015-09-01' order by coin_id"
    basic = pd.read_sql(sql,engine)['coin_id'].tolist()

    sql = "select day,coin_id,close_p from qy_coin_data where close_p <> 0 and day between '2016-01-01' and '2018-08-08' order by coin_id,day"
    df = pd.read_sql(sql,engine)
    df['gold'] = 0
    df['dead'] = 0
    date_list = list(pd.date_range(dt.datetime.strptime('2016-01-01','%Y-%m-%d'),dt.datetime.strptime('2018-08-08','%Y-%m-%d')))
    df_new = pd.DataFrame(columns=df.columns)

    cash = 1000000
    base = cash
    cost = 0.002

    coin_ids = basic[:30]
    coin_pool = {}

    for coin_id in coin_ids:

        df_coin = df[df.coin_id == coin_id]
        df_coin = df_coin.copy()
        if len(df_coin) < len(date_list) - 60:
            continue
        coin_pool.update({coin_id: 0})

        df_coin.index = pd.to_datetime(df_coin['day'])
        df_coin = df_coin.reindex(index=date_list,method='ffill')
        df_coin['coin_id'] = coin_id
        df_coin['MA10'] = df_coin['close_p'].rolling(10).mean()
        df_coin['MA20'] = df_coin['close_p'].rolling(30).mean()
        df_coin['MA10-20'] = df_coin['MA10'] - df_coin['MA20']
        df_coin['gold'] = (df_coin['MA10-20'].shift(1) <= 0) & (df_coin['MA10-20'] > 0)
        df_coin['dead'] = (df_coin['MA10-20'].shift(1) >= 0) & (df_coin['MA10-20'] < 0)
        del df_coin['MA10']
        del df_coin['MA20']
        del df_coin['MA10-20']

        df_new = pd.concat([df_new,df_coin],join='outer')

    df_new['day'] = df_new.index
    df_new.index = df_new['coin_id']
    del df_new['coin_id']

    wealth = [1]

    for date in date_list:
        df_day = df_new[pd.to_datetime(df_new['day']) == date]

        #全部买入、全部卖出
        # if df_coin.iloc[i]['gold'] and coin == 0:
        #     coin = cash / df_coin.iloc[i]['close_p']
        #     cash = 0
        # if df_coin.iloc[i]['dead'] and coin > 0:
        #     cash = coin * df_coin.iloc[i]['close_p'] * (1 - cost)
        #     coin = 0
        marketcap = 0
        for coin_id in coin_pool.keys():
            print(coin_id)
            if df_day.loc[coin_id]['gold'] and cash >= 10000:
                coin_pool[coin_id] = coin_pool[coin_id] + max(cash/30,10000) / df_day.loc[coin_id]['close_p']
                cash = cash - max(cash/30,10000)
            if df_day.loc[coin_id]['dead'] and coin_pool[coin_id] > 0:
                cash = cash + coin_pool[coin_id] * df_day.loc[coin_id]['close_p'] * (1 - cost)
                coin_pool[coin_id] = 0
            marketcap += coin_pool[coin_id] * df_day.loc[coin_id]['close_p']

        total = cash + marketcap

        wealth.append( total / 1000000.)
        # if total / base < 0.95:
        #     cash = cash + coin * df_coin.iloc[i]['close_p'] * (1 - cost)
        #     coin = 0

    print(wealth)
    plt.plot(wealth)
    plt.show()

if __name__ == '__main__':
    main()