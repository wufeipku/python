# - *- coding:utf-8 -*-
import requests
import hashlib
import time

def ticker():
    m = hashlib.md5()
    api_key = '5b7381ae5bf4d'
    api_secret = '702181560e3a3913a98d1d36382ec43605b7381ae'
    dt = time.time()
    params = {'apiKey':api_key,'apiSecrete':api_secret,'timestamp':str(dt),'symbol':'usdt_btc'}
    keys = sorted(params.keys())
    str1 = ''
    for key in keys:
        str1 = str1 + params[key]

    print(str1)
    m.update(str1.encode('gb2312'))
    sign = m.hexdigest()

    param = {'apiKey':api_key,'timestamp':dt,'sign':sign,'symbol':'usdt_btc'}
    re = requests.get('https://openapi.digifinex.com/v2/depth',params=param)
    print(re.json())

if __name__ == '__main__':
    ticker()