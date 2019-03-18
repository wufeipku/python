import requests
import time
from database import engine_local,engine_test
import pandas as pd
import schedule as sd
import datetime as dt
engine = engine_local()

def coin_info():
    re = requests.get('https://api.bitfinex.com/v1/symbols')
    symbols = re.json()
    coin_usd = []
    for i in symbols:
        if i[-3:] == 'usd':
            coin_usd.append('t' + str.upper(i))

    return coin_usd

def tickdata():
    list = coin_info()
    for i in list:
        re = requests.get('https://api.bitfinex.com/v1/tickers',params={'symbols':coin_info()})

    return re.json()

def kline():
    url = 'https://api.bitfinex.com/v2/candles/trade'
    period = ':1m'
    coins = coin_info()
    section = 'hist'

    for coin in coins:
        symbol = ':' + coin
        re = requests.get(url+period+symbol+'/'+section,params={'limit':5})
        re = re.json()
    #coin_ids = coin_info()
        for line in re:
            print(line)
            if line != 'error':
                date = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(line[0]/1000))
                sql = '''replace into kline_1m(coin_id,date,exchange,open,close,high,low,vol) 
                values('{}','{}','bitfinex',{},{},{},{},{}) 
                '''.format(symbol[2:],date,line[1],line[2],line[3],line[4],line[5])
                try:
                    engine.execute(sql)
                except Exception as e:
                    print(e)

    return 0

if __name__ == '__main__':
    # sd.every(30).seconds.do(kline)
    #
    # while True:
    #     sd.run_pending()
    #     time.sleep(1)
    kline()
#校对币名
    # re = requests.get('https://api.bitfinex.com/v1/symbols')
    # symbols = re.json()
    # coin = []
    # for pair in symbols:
    #     coin.append(str.upper(pair[:3]))
    #
    # coin_set = set(coin)
    # print(len(coin_set))
    # da = pd.read_csv('d:/habo/data/chongfu_symbol1.csv')
    # sym = set(da['symbol'].tolist())
    # same = coin_set & sym
    #
    # info = pd.read_sql('select name,symbol from qy_coin_info',engine_test())
    # info.index = info['name']
    #
    # symbol_name = []
    # for symbol in coin_set:
    #     name_ = list(info[info['symbol'] == symbol].index)
    #     if name_:
    #         if len(name_) == 1:
    #             sql = '''insert into exchange_currency(exchange,date,name,symbol) values('Bitfinex','{}','{}','{}')
    #             '''.format(dt.date.today(),name_[0],symbol)
    #             engine.execute(sql)
    #         else:
    #             print({'name':name_,'symbol':symbol})
    #     else:
    #         print(symbol)
    # #print(same)

