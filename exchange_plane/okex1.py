#-*- coding:utf-8 -*-
import websocket
import zlib
import json
import sys
import pandas as pd
import time
from database import engine_local
from multiprocessing import process

# df = pd.read_csv('okex_pairs_increment.csv')
# pair = df['symbol'].tolist()
# engine = engine_local()
# pair = ['eth_btc']
def on_open(self):

    #for i in range(len(pair)):
    #self.send(str({'event':'addChannel','channel':'ok_sub_spot_'+'btc_usdt'+'kline_1min'}))
    #self.send("{'event':'addChannel','channel':'ok_sub_spot_bt2_btc_ticker'}")
    self.send(json.dumps({'method':'subscribeTicker','params':{'symbol':'ZRXBTC'}, 'id':123}))

def on_message(self,message): #data decompress
    print(message)
    # message = json.loads(message)
    # if message[0]['channel'] == 'addChannel':
    #     return
    # pair = message[0]['channel'].split('_')
    # p = str.upper(pair[3]+'_'+pair[4])
    # date = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(int(message[0]['data'][0][0])/1000))
    # sql = "replace into kline_1m(coin_id,date,exchange,open,close,high,low,vol) values('{}','{}','OKEX',{},{},{},{},{})".\
    #     format(p,date,message[0]['data'][0][1],message[0]['data'][0][4],message[0]['data'][0][2],message[0]['data'][0][3],message[0]['data'][0][5])
    # try:
    #     engine.execute(sql)
    # except Exception as e:
    #     print(e)


def on_error(self,error):
    print (error)

def on_close(self):
    print ('DISCONNECT')

if __name__ == '__main__':
    websocket.enableTrace(True)
    #ws = websocket.create_connection("wss://real.okex.com:10440/websocket/okexapi")
    #     # ws.send("{'event': 'addChannel', 'channel': 'ok_sub_futureusd_btc_depth_this_week_5'}")
    #     # print("Receiving...")
    url = "wss://api.hitbtc.com/api/2/ws"
    #websocket.enableTrace(True)
    if len(sys.argv) < 2:
        host = url
    else:
        host = sys.argv[1]

    ws = websocket.WebSocketApp(
        host,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close,
    )


    ws.on_open = on_open

    ws.run_forever()
