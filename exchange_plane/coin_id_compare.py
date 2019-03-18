# -*-coding:utf-8-*-
import pandas as pd
import requests
import re
from database import engine_test

engine = engine_test()

def okex():
    df = pd.read_csv('okex_pairs_increment.csv')
    pairs = df['symbol'].tolist()
    symbol = []
    for pair in pairs:
        s = re.search(r'\w*_',pair)
        symbol.append(s.group()[:-1].upper())

    symbol = set(symbol)
    print(symbol)

    #sql = "SELECT COUNT(id) AS num,name,symbol FROM qy_coin_info WHERE num = 1 GROUP BY symbol"
    sql = "select name,symbol from qy_coin_info"
    df = pd.read_sql(sql,engine)
    df.index = df['name']
    del df['name']
    #del df['num']

    comp = df[df['symbol'].isin(symbol)]
    comp.sort_values(by='symbol',ascending=True,inplace=True)
    print('okex币种数量：%d'% len(symbol))
    print('匹配的数量： %d' % len(comp))
    print(comp)
    comp.to_csv('d:/habo/data/okex.csv')
    return

if __name__ == '__main__':
    okex()