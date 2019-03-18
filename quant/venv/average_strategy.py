from database import engine_test
import pandas as pd

def data_clean():
    coin_ids = (1, 11, 7, 9, 1343, 3, 1347, 4, 1341)
    df = pd.read_sql(
        'select a.coin_id,a.day,a.close_p,b.symbol from qy_coin_data a left join qy_coin_info b on a.coin_id = b.id where a.day >= "2017-01-01" and a.coin_id in {} order by a.coin_id,a.day'.format(coin_ids),
        engine_test())
    df_new = pd.DataFrame()
    
    for coin_id in coin_ids:
        df_coin = df[df['coin_id'] == coin_id]
        date_list = pd.date_range(df_coin.iloc[0]['day'], '2018-12-05')
        df_coin.index = pd.to_datetime(df_coin.day)
        df_coin = df_coin.reindex(index=date_list, method='ffill')
        df_new = pd.concat([df_new, df_coin], join='outer')

    return df_new

def trade(df_new, cash=1000000):
    symbols = set(df_new['symbol'])
    component = {}
    component_list = []
    wealth = [cash]
    date_list = pd.date_range('2017-01-01','2018-12-05')
    
    for symbol in symbols:
        component.update({symbol:[0,0,0]}) #分别记录持仓量和持仓市值

    df_day = df_new.loc[date_list[0]]
    #l_before = len(df_day)
    
    for coin_id in df_day['symbol']:
        cap = cash / len(df_day)
        volume = cap / df_day[df_day['symbol']==coin_id].iloc[0]['close_p']
        component[coin_id] = [volume, cap, 0]
    
    component_list.append(component)

    for date in date_list[1:]:
        df_day = df_new.loc[date]
        component_new = {}
        l = 0 #记录持仓币数
        temp = 0

        for key in component:
            if component[key][0] > 0:
                volume = component[key][0]
                cap = component[key][0] * df_day[df_day.symbol==key].iloc[0]['close_p']
                change = cap - component[key][1]
                component_new[key] = [volume, cap, change]
                temp += component_new[key][1]
                l = l + 1
            else:
                component_new[key] = [0,0,0]

        wealth.append(temp)
        if l < len(df_day):
            for symbol in df_day['symbol']:
                cap = wealth[-1] / len(df_day)
                volume = cap / df_day[df_day['symbol'] == symbol].iloc[0]['close_p']
                if component_new[symbol][0] > 0:
                    component_new[symbol][0] = volume
                    component_new[symbol][1] = cap
                else:
                    component_new[symbol] = [volume, cap, 0]
        
        component_list.append(component_new)
        component = component_new
        #print(component_list[0]['BTC'])


    #print(pd.DataFrame(component_list))

    column = ['wealth','return']
    column.extend(list(component.keys()))
    result = pd.DataFrame(index=date_list,columns=column)
    result.wealth = wealth
    result['return'] = result.wealth / cash - 1


    for key in component:
        collection = []
        for one in component_list:
            collection.append(one[key][1])
        result[key] = collection
    #result.fillna(0, inplace=True)

    return result

if __name__ == '__main__':
    df_new = data_clean()
    #print(df_new)
    data = trade(df_new)
    data.to_csv('d:/average_strategy_1.csv')
    #print(data)
    


