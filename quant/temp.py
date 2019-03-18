import pandas as pd
from database import engine_test,engine_foreign
from matplotlib import pyplot as plt
import numpy as np
from math import ceil
import hashlib
import time
import datetime as dt
import requests

if __name__ == '__main__':
#涨跌幅分布
    # df = pd.read_sql('select a.id,a.hour24 from qy_coin_info a left join qy_coin_data b on a.id = b.coin_id and b.day = "2018-09-09" order by market_cap desc limit 500',engine_test())
    # #df = pd.read_sql('select id,hour24 from qy_coin_info',engine_test())
    # #print(df)
    # df = df[~df['hour24'].isna()]
    # data = df['hour24'].tolist()
    # pdata = [0] * 22
    # for a in data:
    #     if a >= 10:
    #         pdata[21] += 1
    #     elif a <= -10:
    #         pdata[0] += 1
    #     else:
    #         pdata[ceil(a)+10] += 1
    # print(pdata)
    # plt.bar(range(22),pdata)
    # #plt.ylim(0,100)
    # plt.show()
    # #print(len(df[ df['hour24'] < 0]),len(df[ df['hour24'] < -1]))

    # df = pd.read_sql('SELECT last_updated,price_usd FROM qy_coin_data_detail WHERE info_id = 1555 AND last_updated < UNIX_TIMESTAMP(NOW())AND last_updated >= UNIX_TIMESTAMP(NOW()) - 24*3600 ORDER BY last_updated',engine_test())
    # print(len(df))
    # plt.plot(df['last_updated'],df['price_usd'])
    # plt.show()

    # df = pd.read_sql('select distinct(name),symbol from qy_exch_coin',engine_test())
    # df_info = pd.read_sql('select name,symbol from qy_coin_info a LEFT JOIN qy_coin_data b ON a.id = b.coin_id WHERE b.day = "2018-09-10" ORDER BY b.market_cap DESC',engine_test())
    #
    # # print(df_info.iloc[0]['name'].replace(' ','').upper() == 'MCO')
    # # print(df.iloc[0]['symbol'] == df_info.iloc[0]['symbol'])
    # name1 = df['name'].tolist()
    # symbol1 = df['symbol'].tolist()
    # name2 = df_info['name'].tolist()
    #
    # new = []
    # for i in range(len(name1)):
    #     name1[i] = name1[i].replace(' ','').upper()
    #     df_new = df_info[df_info['symbol'] == symbol1[i]]['name'].tolist()
    #     if len(df_new) > 0:
    #         for name in df_new:
    #             if (name1[i] in name.replace(' ','').upper()) or (name.replace(' ','').upper() in name1[i]) \
    #                     :
    #                 new.append(name)
    #                 break
    #
    # print(len(new),len(df_info))
    # print(new)
    # print(set(name2) - set(new))

    # coin_ids = (1, 11, 12, 10, 7, 1360, 9, 1343, 8, 3, 1347, 6, 4, 1341, 2)
    # sql = "select * from qy_coin_data where coin_id in {} and day >= '2017-01-01'".format(coin_ids)
    # df = pd.read_sql(sql, engine_test())
    # data = pd.DataFrame(index=coin_ids,columns=['return','day'])
    # for id in coin_ids:
    #     df_coin = df[df.coin_id == id]
    #     df_coin.sort_values(by='day',ascending=True,inplace=True)
    #     re = df_coin.close_p.tolist()[-1] / df_coin.close_p.tolist()[0] - 1
    #     data.loc[id, 'return'] = re
    #     data.loc[id, 'day'] = df_coin.iloc[0]['day']
    #
    # print(data)
#serchchain数据对接----------------------------------------------------------------------------------------------------
    # apiKey = '184dd16ea3f6b05c6469aff3555a2911'
    # apiSecret = 'lQsYDhVyLSp3XLDhyOB9zWfkXcpSHqunVkax5ggdttk='
    # localtime = int(time.time())
    # param = '&timestamp={}'.format(localtime) + apiSecret
    # sign = hashlib.sha1(param.encode(encoding='utf8')).hexdigest()
    # result = apiKey + ':'+ sign + ':' + str(localtime)
    # header = {'Accept': 'application/vnd.searchain.v1+json', 'Authorization': result}
    # re = requests.get('http://api.searchain.io/coins/all', headers=header)
    # print(len(re.json()))
#-----------------------------------------------------------------------------------------------------------------------
#https://api.binance.com/api/v1/klines?symbol=ETHBTC&interval=1d
    # con = requests.get('https://graphs2.coinmarketcap.com/global/marketcap-total/')
    # data_js = con.json()
    # data = data_js['market_cap_by_available_supply']
    # data = pd.DataFrame(data, columns=['date','marketcap'])
    # def stamp2date(stamp):
    #     t = time.localtime(stamp/1000)
    #     date = dt.datetime(t.tm_year,t.tm_mon,t.tm_mday)
    #     return date
    #
    # data['date'] = data['date'].apply(stamp2date)
    # data.index = data['date']
    # del data['date']
    # print(data)
    # data.to_excel('d:/history_marketcap.xlsx')
#--------------------------------------------------------------------
    df = pd.read_sql('select distinct(pair) as p from qy_exch_ticker where exchange = "HitBTC"', engine_foreign())
    df['p'] = df['p'].apply(lambda x:x.replace('_', ''))
    da = df['p'].tolist()
    re = requests.get('https://api.hitbtc.com/api/2/public/symbol')
    re = re.json()
    data = []

    for d in re:
        if d['id'] not in da:
            data.append(d['id'])

    print(len(data))
    print(tuple(data))



