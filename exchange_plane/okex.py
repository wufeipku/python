#-*- coding:utf-8 -*-
#application/x-www-form-urlencoded
import requests
import pandas as pd
from database import engine_local
import time
df = pd.read_csv('okex_pairs_increment.csv')
pairs = df['symbol'].tolist()

def kline_1m():

    url = 'https://www.okex.com/api/v1/kline.do'

    #for pair in pairs[0]:
    #i = 0
    while 1:
        time1 = time.time()
        for i in range(len(pairs)):

            re = requests.get(url,params={'symbol':pairs[i],'type':'1min'})
            #re = requests.get('https://www.okex.com/api/v1/trades.do?symbol=bch_btc')
            re = re.json()
            #print(re[-1]['date'] - re[0]['date'])
            print(pairs[i],len(re),re)

        print(time.time() - time1,'********************************************************************')

if __name__ == '__main__':
    kline_1m()
