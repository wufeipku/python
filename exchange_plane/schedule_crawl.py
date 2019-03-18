import schedule as sd
import time
import datetime as dt
from api import OKex
import threading
import requests
import numba

def get_exchange():
    dt = OKex()
    data = dt.dollar_rmb()
    data.update({'time':dt.datetime.now()})
    print(data)

def get_trade():
    dt = OKex()
    data = dt.trades('btc_usd','this_week')
    print(data)


def get_exchange_copy():
    threading.Thread(target=get_exchange).start()

def get_trade_copy():
    threading.Thread(target=get_trade).start()


def Bnet_pair():
    url_pair = 'https://bittrex.com/api/v1.1/public/getmarkets'
    pair_ = requests.get(url_pair)
    pair_ = pair_.json()
    pair_ = pair_['result']

    pair = []
    for p in pair_:
        pair.append(p['MarketName'])

    return pair

def Bnet_ticker():
    pair = Bnet_pair()
    url_ticker =  'https://bittrex.com/api/v1.1/public/getticker'

    result = []
    for p in pair:
        ticker = requests.get(url_ticker,params={'market':p})
        ticker = ticker.json()
        result.append([dt.datetime.now(),p,ticker['result']['Last']])

    print(result)

    return

if __name__ == '__main__':
    pair = Bnet_pair()
    sd.every(10).seconds.do(Bnet_ticker(pair))
    #sd.every(10).seconds.do(get_trade_copy)
    while True:
        sd.run_pending()
        time.sleep(1)

