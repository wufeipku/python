#-*- coding:utf-8 -*-
import websocket
import json
import sys
import pandas as pd
import time
from database import engine_local
import requests

engine = engine_local()
re = requests.get('https://api.binance.com/api/v1/exchangeInfo')
data = re.json()['symbols']
pairs = []
for d in data:
    pairs.append(d['symbol'])

pair_dict = {}
initial = {}
def on_open(self):

    for p in pairs[0:1]:

        self.send('{"stream":"bnbbtc@depth","data":"message"}')
        #self.send("{'event':'addChannel','channel':'ok_sub_spot_abl_btc_ticker'}")
def on_message(self,message): #data decompress
    message = json.loads(message)
    print(message)
#-----------------------------------------------------------------------------------------
    #k线数据存储
    # print(pair_dict)
    # if len(message) == 4:
    #     if message['event'] == 'subscribed':
    #         pair_dict.update({message['chanId'] : message['key'][10:]})
    #
    #     return
    # elif len(message) == 2:
    #     if len(message[1]) == 6:
    #         pair = pair_dict[message[0]][0:3]+'_'+ pair_dict[message[0]][3:]
    #         date = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(int(message[1][0])/1000))
    #         sql = "replace into kline_1m(coin_id,date,exchange,open,close,high,low,vol) values('{}','{}','BitFinex',{},{},{},{},{})".\
    # format(pair,date,message[1][1],message[1][2],message[1][3],message[1][4],message[1][5])
    #         try:
    #             engine.execute(sql)
    #         except Exception as e:
    #             print(e)
#-----------------------------------------------------------------------------------------
    #逐笔交易数据存储
    # if len(message) != 3 and len(message) != 2:
    #     if message['event'] == 'subscribed':
    #         pair_dict.update({message['chanId']:message['pair']})
    # elif len(message) == 3:
    #     if message[1] == 'tu':
    #         pair = pair_dict[message[0]][:3] + '_' + pair_dict[message[0]][3:]
    #         sql = "replace into trades(coin_id,pair,exchange,date,volume,price) values('{}','{}','BitFinex',{},{},{})". \
    #             format(pair[:3],pair,message[2][1],message[2][2],message[2][3])
    #         try:
    #             engine.execute(sql)
    #         except Exception as e:
    #             print(e)
#-----------------------------------------------------------------------------------------
# #ticker数据
#     if len(message) != 2:
#         if message['event'] == 'subscribed':
#             pair_dict.update({message['chanId']:message['pair']})
#     elif message[1] != 'hb':
#         pair = pair_dict[message[0]][:3] + '_' + pair_dict[message[0]][3:]
#         date = int(time.time() * 1000)
#         if pair[4:] in ['BTH','EOS']:
#             df = pd.read_sql('select last_price from ticker where pair="{}" order by date desc limit 1'.format(pair[4:]+'_BTC'),engine)
#             last_price = message[1][6] * df.iloc[0][0]
#             change_amount_24h = message[1][4] * df.iloc[0][0]
#             high = message[1][8] * df.loc[0][0]
#             low = message[1][9] * df.loc[0][0]
#         else:
#             last_price = message[1][6]
#             change_amount_24h = message[1][4]
#             high = message[1][8]
#             low = message[1][9]
#         sql = "replace into ticker(coin_id,pair,exchange,date,last_price,volume_24h,turnover_24h,change_24h,change_amount_24h,high_24h,low_24h) values('{}','{}','BitFinex',{},{},{},{},{},{},{},{})". \
#             format(pair[:3],pair,date,last_price,message[1][7],0,message[1][5],change_amount_24h,high,low)
#         try:
#             engine.execute(sql)
#         except Exception as e:
#             print(e)
#-----------------------------------------------------------------------------------------
    #depth数据







def on_error(self,error):
    print (error)

def on_close(self):
    print ('DISCONNECT')

if __name__ == '__main__':
    # ws = websocket.create_connection("wss://real.okex.com:10440/websocket/okexapi")
    #     # ws.send("{'event': 'addChannel', 'channel': 'ok_sub_futureusd_btc_depth_this_week_5'}")
    #     # print("Receiving...")
    url = "wss://stream.binance.com:9443/ws/bnbbtc@depth"
    #url = 'wss://real.okex.com:10441/websocket'
    # websocket.enableTrace(True)
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
