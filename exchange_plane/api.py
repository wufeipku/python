#  -*- coding: utf-8 -*-

import pandas as pd
import requests
import json
import time


class OKex():
    def __init__(self):
        self.okex_url = {
            '行情': 'https://www.okex.com/api/v1/future_ticker.do',
            '深度': 'https://www.okex.com/api/v1/future_depth.do',
            '交易': 'https://www.okex.com/api/v1/future_trades.do',
            '指数': 'https://www.okex.com/api/v1/future_index.do',
            '汇率': 'https://www.okex.com/api/v1/exchange_rate.do',
            '交割预估价': 'https://www.okex.com/api/v1/future_estimated_price.do',
            'k线': 'https://www.okex.com/api/v1/future_kline.do'
        }

        self.url = self.okex_url

    def tiker(self,symbol=None,contract_type=None):
        data = requests.get(self.url['行情'],params={'symbol':symbol,'contract_type':contract_type})
        data = data.json()

        return data

    def depth(self,symbol=None,contract_type=None,size=None,merge=None):
        data = requests.get(self.url['深度'], params={'symbol':symbol,'contract_type':contract_type,'size':size,'merge':merge})
        data = data.json()

        return data

    def trades(self,symbol=None,contract_type=None):
        data = requests.get(self.url['交易'],params={'symbol':symbol,'contract_type':contract_type})
        data = data.json()

        return data

    def index(self,symbol=None):
        data = requests.get(self.url['指数'],params={'symbol':symbol})
        data = data.json()

        return data

    def dollar_rmb(self):
        data = requests.get(self.url['汇率'])
        data =data.json()

        return data

    def k_line(self,symbol=None,type=None,contract_type=None,size=None,since=None):
        '''
        :param symbol:btc_usd   ltc_usd    eth_usd    etc_usd    bch_usd
        :param type:1min/3min/5min/15min/30min/1day/3day/1week/1hour/2hour/4hour/6hour/12hour
        :param contract_type:合约类型: this\_week:当周   next\_week:下周   quarter:季度
        :param size: 获取数据条数，默认0
        :param since: 开始时间戳，默认0
        :return:
        '''
        data = requests.get(self.url['k线'],params={'symbol':symbol,'type':type,'contract_type':contract_type,'size':size,'since':since})
        data = data.json()

        return data

# class bnb():
#     def __init__(self):
#
#
#     def kline(self,symbol=None,interval='1d',startTime=None,endTime=None,limit=1000):
#         param = {
#             'symbol':symbol,
#             'interval':interval,
#             'startTime':startTime,
#             'endTime':endTime,
#             'limit':limit
#         }
#         re = requests.get('https://api.binance.com/api/v1/klines',params=param)
#         re = re.json()
#
#         return re


if __name__ == '__main__':
    dt = OKex()
    '''
    params = {
        'symbol':'btc_usd' 'ltc_usd' 'eth_usd' 'etc_usd' 'bch_usd'
        'contract_type':'this week','next_week','quarter'
        'size':1-200
        'merge':0
    }
    '''


    # tiker_data = dt.tiker('btc_usd','next_week')
    # print(time.strftime('%Y%m%d %H:%M:%S',time.localtime(int(tiker_data['date']))),tiker_data['ticker']['last'])
    # #print(tiker_data)
    # trade_data = dt.trades('btc_usd','this_week')
    # da = pd.DataFrame(trade_data)
    # da['date'] = da['date'].apply(lambda x:time.strftime('%Y%m%d %H:%M:%S',time.localtime(x)))
    # print(da[['date','price']])
    #
    #index_data = dt.index('btc_usd')
    #da = pd.DataFrame(index_data)

    # exchange_rate = dt.dollar_rmb()
    # print(exchange_rate)
    k_line = dt.k_line('btc_usd','1min','next_week')
    da = pd.DataFrame(k_line,columns=['time','open','high','low','close','volume','to_btc'])
    da['time'] = da['time'].apply(lambda x: time.strftime('%Y%m%d %H:%M:%S',time.localtime(x/1000)))
    print(da)
