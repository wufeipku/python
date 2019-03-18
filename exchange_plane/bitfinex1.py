#-*- coding:utf-8 -*-
import websocket
import json
import sys
import pandas as pd
import time
from database import engine_local
import requests

engine = engine_local()
re = requests.get('https://api.bitfinex.com/v1/symbols')
pairs = re.json()
pair_dict = {}
initial = {}
def on_open(self):

    for p in pairs:
        #self.send(json.dumps({'event':'subscribe','channel':'candles','key':'trade:1m:t'+str.upper(p)}))
        #self.send(json.dumps({'event':'subscribe','channel':'trades','symbol':'t'+ str.upper(p)}))
        self.send(json.dumps({'event':'subscribe','channel':'ticker','symbol':'t'+ str.upper(p)}))
        #self.send(json.dumps({'event':'subscribe','channel':'book','symbol':'t'+ str.upper(p),'len':25}))
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
    # if message is dict:
    #     if message['event'] == 'subscribed':
    #          pair_dict.update({message['chanId']:message['pair'][:3]+'_'+message['pair'][3:]})
    # elif len(message[1]) != 3:
    #     pair = pair_dict[message[0]]
    #     initial.update({message[0]:message[1]})
    #     print(pair)
    #     ask = message[1][25:35]
    #     bid = message[1][:10]
    #     date = int(time.time() * 1000)
    #     sql = "replace into depth values('{}','{}','BitFinex',{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{})". \
    #                 format(pair[:3],pair,date,ask[0][0],ask[1][0],ask[2][0],ask[3][0],ask[4][0],ask[5][0],ask[6][0],ask[7][0],ask[8][0],ask[9][0],
    #                        ask[0][2], ask[1][2], ask[2][2], ask[3][2], ask[4][2], ask[5][2], ask[6][2], ask[7][2],ask[8][2], ask[9][2],
    #                        bid[0][0],bid[1][0],bid[2][0],bid[3][0],bid[4][0],bid[5][0],bid[6][0],bid[7][0],bid[8][0],bid[9][0],
    #                        bid[0][2], bid[1][2], bid[2][2], bid[3][2], bid[4][2], bid[5][2], bid[6][2], bid[7][2],
    #                        bid[8][2], bid[9][2])
    #     try:
    #         engine.execute(sql)
    #     except Exception as e:
    #         print(e)
    # else:
    #     data = initial[message[0]]
    #     ask = data[25:]
    #     bid = data[:25]
    #     if message[1][2] > 0:
    #         for i in range(len(bid)):
    #             if bid[i][0] < message[1][0]:
    #                 if message[1][1] != 0:
    #                     try:
    #                         bid[i+1:] = bid[i:-1]
    #                         bid[i] = message[1]
    #                     except:
    #                         print('go on')
    #                 break
    #             elif bid[i][0] == message[1][0]:
    #                 if message[1][1] != 0:
    #                     bid[i] = message[1]
    #                 else:
    #                     bid[i:-1] = bid[i+1:]
    #                     bid[-1] = [0,0,1]
    #                 break
    #     else:
    #         for i in range(len(ask)):
    #             if ask[i] > message[1][0]:
    #                 if message[1][1] != 0:
    #                     try:
    #                         ask[i+1:] = ask[i:-1]
    #                         ask[i] = message[1]
    #                     except:
    #                         print('go on')
    #             elif ask[i] == message[1][0]:
    #                 if message[1][1] != 0:
    #                     ask[i] = message[1]
    #                 else:
    #                     ask[i:-1] = ask[i + 1:]
    #                     ask[-1] = [0,0,-1]
    #                 break
    #     initial[message[0]] = [bid+ask]
    #
    #     pair = pair_dict[message[0]]
    #     date = int(time.time() * 1000)
    #     sql = "replace into depth values('{}','{}','BitFinex',{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{})". \
    #         format(pair[:3], pair, date, ask[0][0],ask[1][0],ask[2][0],ask[3][0],ask[4][0],ask[5][0],ask[6][0],ask[7][0],ask[8][0],ask[9][0],
    #                        ask[0][2], ask[1][2], ask[2][2], ask[3][2], ask[4][2], ask[5][2], ask[6][2], ask[7][2],ask[8][2], ask[9][2],
    #                        bid[0][0],bid[1][0],bid[2][0],bid[3][0],bid[4][0],bid[5][0],bid[6][0],bid[7][0],bid[8][0],bid[9][0],
    #                        bid[0][2], bid[1][2], bid[2][2], bid[3][2], bid[4][2], bid[5][2], bid[6][2], bid[7][2],
    #                        bid[8][2], bid[9][2])
    #     try:
    #         engine.execute(sql)
    #     except Exception as e:
    #         print(e)






def on_error(self,error):
    print (error)

def on_close(self):
    print ('DISCONNECT')

if __name__ == '__main__':
    # ws = websocket.create_connection("wss://real.okex.com:10440/websocket/okexapi")
    #     # ws.send("{'event': 'addChannel', 'channel': 'ok_sub_futureusd_btc_depth_this_week_5'}")
    #     # print("Receiving...")
    url = "wss://api.bitfinex.com/ws/2"
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

