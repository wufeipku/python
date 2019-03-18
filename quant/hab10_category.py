#陈亮
from database import engine_test
import pandas as pd


def data_clean():
    coin_ids = (1, 3, 4, 7, 9, 11, 12, 1341, 1347, 1360, 2915)
    df = pd.read_sql(
        'select a.coin_id,a.day,a.close_p,a.market_cap, b.symbol from qy_coin_data a left join qy_coin_info b on a.coin_id = b.id where a.day >= "2017-01-01" and a.coin_id in {} order by a.coin_id,a.day'.format(
            coin_ids),
        engine_test())
    df_new = pd.DataFrame()

    df_btc = df[df['coin_id'] == 1]
    date_list = pd.date_range(df_btc.iloc[0]['day'], '2018-12-09')
    df_btc.index = pd.to_datetime(df_btc.day)
    df_btc = df_btc.reindex(index=date_list, method='ffill')

    for coin_id in coin_ids:
        if coin_id == 1: continue
        df_coin = df[df['coin_id'] == coin_id]
        date_list = pd.date_range(df_coin.iloc[0]['day'], '2018-12-09')
        df_coin.index = pd.to_datetime(df_coin.day)
        df_coin = df_coin.reindex(index=date_list, method='ffill')
        btc_price = df_btc[date_list[0]:date_list[-1]]['close_p']
        df_coin = df_coin.copy()
        df_coin['price_btc'] = df_coin['close_p'] / btc_price
        df_new = pd.concat([df_new, df_coin], join='outer')

    return df_new


def trade(df_new, cash=1000000, uplimit=0.25):
    symbols = set(df_new['symbol'])
    component = {}
    component_list = []
    wealth = [cash]
    date_list = pd.date_range('2018-09-10', '2018-12-09')

    for symbol in symbols:
        component.update({symbol: [0, 0, 0]})  # 分别记录持仓量和持仓市值
    #初期配置------------------------------------------------------------
    df_day = df_new.loc[date_list[0]]
    df_day = df_day.copy()
    df_day.sort_values(by='market_cap', ascending=False, inplace=True)
    #marketcap = df_day['market_cap'].sum()
    percent = [0.0] * len(df_day)

    for i in range(len(df_day)):
        percent[i] = (1 - sum(percent[0:i])) * df_day.iloc[i]['market_cap'] / df_day['market_cap'][i:].sum()
        if percent[i] > uplimit:
            percent[i] = uplimit
        cap = cash * percent[i]
        volume = cap / df_day.iloc[i]['close_p']
        component[df_day.iloc[i]['symbol']] = [volume, cap, 0]

    component_list.append(component)
#-----------------------------------------------------------------------------------------------
    for date in date_list[1:]:
        #fee = 0 #手续费
        df_day = df_new.loc[date]
        component_new = {}
        l = 0  # 记录持仓币数
        temp = 0

        for key in component:
            if component[key][0] > 0:
                volume = component[key][0]
                cap = component[key][0] * df_day[df_day.symbol == key].iloc[0]['close_p']
                change = cap - component[key][1]
                component_new[key] = [volume, cap, change]
                temp += component_new[key][1]
                l = l + 1
            else:
                component_new[key] = [0, 0, 0]

        wealth.append(temp)
        #有新币增加的情况判断，新币增加则重新按市值分配比例-------------------------------------------------------------
        if l < len(df_day):
            #marketcap = df_day['market_cap'].sum()
            percent = [0.0] * len(df_day)
            df_day = df_day.copy()
            df_day.sort_values(by='market_cap', ascending=False, inplace=True)

            for i in range(len(df_day)):
                percent[i] = (1 - sum(percent[0:i])) * df_day.iloc[i]['market_cap'] / df_day['market_cap'][i:].sum()
                if percent[i] > uplimit:
                    percent[i] = uplimit
                cap = wealth[-1] * percent[i]
                volume = cap / df_day.iloc[i]['close_p']
                symbol = df_day.iloc[i]['symbol']
                if component_new[symbol][0] > 0:
                    component_new[symbol][0:2] = [volume, cap]
                else:
                    component_new[symbol] = [volume, cap, 0]
        #没有新币增加，则判断占比是否满足上限限制----------------------------------------------------------------------
        # else:
        #     marketcap_list = pd.DataFrame(component_new, index=['volume', 'cap', 'change'])
        #     marketcap_list = marketcap_list.T
        #     marketcap_list = marketcap_list[marketcap_list['volume']>0]
        #     marketcap_list.sort_values(by='cap', ascending=False, inplace=True)
        #     percent = [0.0] * len(df_day)
        #
        #     for i in range(len(marketcap_list)):
        #         percent[i] = (1 - sum(percent[0:i])) * marketcap_list.iloc[i]['cap'] / marketcap_list['cap'][i:].sum()
        #         if percent[i] > uplimit:
        #             percent[i] = uplimit
        #         cap = wealth[-1] * percent[i]
        #         volume = cap / marketcap_list.iloc[i]['price_btc']
        #         symbol = marketcap_list.iloc[i]['symbol']
        #         outcome = 0
        #         if volume < component_new[symbol][0]:
        #             outcome = (component_new[symbol][0] - volume) * df_day[df_day['symbol']==symbol].iloc[0]['price_btc'] * cost
        #             fee += outcome
        #             component_new[symbol][2] = component_new[symbol][2] - outcome
        #         component_new[symbol][0:2] = [volume, cap]

        component_list.append(component_new)
        component = component_new
        # print(component_list[0]['BTC'])

    # print(pd.DataFrame(component_list))

    column = ['wealth', 'return']
    column.extend(list(component.keys()))
    result = pd.DataFrame(index=date_list, columns=column)
    result.wealth = wealth
    result['return'] = result.wealth / cash - 1

    for key in component:
        collection = []
        for one in component_list:
            collection.append(one[key][1])
        result[key] = collection
    # result.fillna(0, inplace=True)

    return result


if __name__ == '__main__':
    df_new = data_clean()
    # print(df_new)
    flag = -10000
    param = 0.1
    result = []

    # for num in range(10, 26, 1):
    #     uplimit = num / 100.0
    #     data = trade(df_new, uplimit=uplimit)
    #     profit = data['return'].tolist()[-1]
    #     result.append([profit,uplimit])
    #     print(result[-1])
    #     if profit > flag:
    #         flag = profit
    #         param = uplimit
    #
    # data = pd.DataFrame(result, columns=['return', 'uplimit'])
    # print('(uplimit, return) is (%f,%f)' % (param,flag))
    # data.to_csv('d:/hab10_近3个月.csv')
    # print(data)
    for uplimit in [0.17, 0.25]:
        data = trade(df_new, uplimit=uplimit)
        #data.to_csv('d:/hab10_近1年_uplimit{}.csv'.format(int(uplimit * 100)))
        data.to_excel('d:/hab10_近3个月_uplimit{}.xlsx'.format(int(uplimit * 100)))



