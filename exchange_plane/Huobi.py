# -*- coding: utf-8 -*-
#author: 半熟的韭菜

from websocket import create_connection
import gzip
import time
import json

if __name__ == '__main__':
    while(1):
        try:
            ws = create_connection("wss://api.hitbtc.com/api/2/ws")
            break
        except:
            print('connect ws error,retry...')
            time.sleep(5)

    # 订阅 KLine 数据
    #tradeStr="""{"sub": "market.tickers","id": "id10"}"""

    # 请求 KLine 数据
    # tradeStr="""{"req": "market.ethusdt.kline.1min","id": "id10", "from": 1513391453, "to": 1513392453}"""

    #订阅 Market Depth 数据
    # tradeStr="""{"sub": "market.ethusdt.depth.step5", "id": "id10"}"""

    #请求 Market Depth 数据
    # tradeStr="""{"req": "market.ethusdt.depth.step5", "id": "id10"}"""

    #订阅 Trade Detail 数据
    tradeStr="{'method':'subscribeTicker','params':{'symbol':'ETHBTC'}, 'id':123}"

    #请求 Trade Detail 数据
    # tradeStr="""{"req": "market.ethusdt.trade.detail", "id": "id10"}"""

    #请求 Market Detail 数据
    # tradeStr="""{"req": "market.ethusdt.detail", "id": "id12"}"""
   # ticker = """{"req": "market.tickers","id":"id10"}"""
    ws.send(tradeStr)
    trade_id = ''
    while(1):
        data = ws.recv()
        print(data)
        # compressData=ws.recv()
        # result=gzip.decompress(compressData).decode('utf-8')
        # if result[:7] == '{"ping"':
        #     ts=result[8:21]
        #     pong='{"pong":'+ts+'}'
        #     ws.send(pong)
        #     ws.send(tradeStr)
        # else:
        #     try:
        #         if trade_id == result['data']['id']:
        #             print('重复的id')
        #             break
        #         else:
        #             trade_id = result['data']['id']
        #     except Exception:
        #         pass
        #
        #     result = json.loads(result)
        #
        #     if 'ch' in result:
        #         data = result['data']
        #         flag = []
        #
        #         for d in data:
        #             if d['symbol'][-3:] == 'btc':
        #                 flag.append(d['symbol'][0:-3])
        #
        #         print(flag)

            #print(result)