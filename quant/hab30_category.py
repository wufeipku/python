# -*- coding:utf-8 -*-

import pandas as pd
import datetime as dt
import numpy as np
from dateutil.relativedelta import relativedelta
import pymysql
from pprint import pprint
import numba
from database import engine_test
from pymongo import MongoClient

# mongo设置
#线上
client = MongoClient('127.0.0.1', 27017)
#测试
#client = MongoClient('10.100.0.113',27017)
#client = MongoClient('127.0.0.1', 27017)
#mongodbUri = 'mongo 172.17.226.30:27017/fof_api_currency'
#client = MongoClient(mongodbUri)
db = client['fof_api_currency']

engine = engine_test()


def component_all():
    # 计算每期成分股
    startdate = '2017-01-01'
    statistic_date = []
    while dt.datetime.strptime(startdate, '%Y-%m-%d') <= dt.datetime.today():
        statistic_date.append(startdate)
        startdate = (dt.datetime.strptime(startdate, '%Y-%m-%d') + relativedelta(months=1)).strftime('%Y-%m-%d')

    component_dict = {}
    for standard_date in statistic_date:
        component = list(coin_component(standard_date))
        if len(component) < 10:
            component.extend([np.nan] * (10 - len(component)))
        component_dict[standard_date] = component

    db.drop_collection('coin_component')
    db.coin_component.insert_one(component_dict)
    # component_pd = pd.DataFrame(component_dict)
    # component_pd.to_csv('component.csv')

    # 计算权重
    weight = {}
    market_date_list = list(pd.date_range(start='2017-01-01', end=dt.datetime.today() - dt.timedelta(1)))
    sql = "select coin_id, day, market_cap from qy_coin_data where day between '{}' and '{}' order by coin_id,day".\
        format(market_date_list[0],market_date_list[-1])
    df = pd.read_sql(sql, engine)
    for i in range(len(statistic_date)):
        if i == 0:
             statistic_date_before = '2017-01-01'
        else:
             statistic_date_before = (dt.datetime.strptime(statistic_date[i], '%Y-%m-%d') - dt.timedelta(1)).strftime('%Y-%m-%d')
        list_marketvalue = []

        for coin_id in component_dict[statistic_date[i]]:
            if coin_id != '':
                df_coin = df[df.coin_id == coin_id]
                df_coin.index = pd.to_datetime(df_coin.day)
                df_coin = df_coin.reindex(index=market_date_list, method='ffill')
                market_value = df_coin.loc[statistic_date_before, 'market_cap']
                list_marketvalue.append([coin_id, market_value, 1.])

        weight[statistic_date[i]] = index_statistics(list_marketvalue)['ratio'].tolist()
        if len(weight[statistic_date[i]]) < 10:
            weight[statistic_date[i]].extend([np.nan] * (10 - len(weight[statistic_date[i]])))

    db.drop_collection('coin_weight')
    db.coin_weight.insert_one(weight)
    # weight = pd.DataFrame(weight)
    # weight.to_csv('weight.csv')


def coin_component(statistic_date):
    """
    成分虚拟币
    :param statistic_date:
    :return component_coin:
    """
    # 整理日期
    statistic_date = dt.datetime.strptime(statistic_date, '%Y-%m-%d') - dt.timedelta(days=1)
    date_stable = str(statistic_date - relativedelta(months=6))
    first_date = [statistic_date - relativedelta(months=i) for i in range(6, 0, -1)]
    last_date = [statistic_date - relativedelta(months=i) - relativedelta(days=1) for i in range(5, -1, -1)]
    date_zip = zip(first_date, last_date)

    # 稳定交易六个月的虚拟币
    sql = "select distinct coin_id,publish from qy_coin_info_extend"
    df = pd.read_sql(sql, engine)
    df = df[df.publish < date_stable]
    component_stable = set(df.coin_id.drop_duplicates().tolist())

    # 过去6个月市值连续超过3个月排名在100名以内
    ranks = []
    for date in date_zip:
        sql = "select coin_id,market_cap,volume from qy_coin_data where day BETWEEN '{}' and '{}' order by day" \
            .format(date[0], date[1])
        df = pd.read_sql(sql, engine)
        coin_ids = df.coin_id.drop_duplicates().tolist()
        market_value_index = df.groupby('coin_id').sum().sort_values(by='market_cap',
                                                                     ascending=False).index.drop_duplicates().tolist()
        turnover_index = df.groupby('coin_id').sum().sort_values(by='volume',
                                                                 ascending=False).index.drop_duplicates().tolist()
        rank_dict = {}

        for coin_id in coin_ids:
            rank_dict[coin_id] = market_value_index.index(coin_id) + turnover_index.index(coin_id)
        rank = [i[0] for i in sorted(rank_dict.items(), key=lambda x: x[1])[:100]]
        ranks.append(set(rank))
    component_rank = (ranks[0] & ranks[1] & ranks[2]) | (ranks[1] & ranks[2] & ranks[3]) | (
            ranks[2] & ranks[3] & ranks[4]) | (ranks[3] & ranks[4] & ranks[5])
    # 分配token和coin的比例
    # 使用我们的库获取分类
    sql = '''SELECT a.id,c.tags,cate,b.market_cap 
              FROM qy_coin_info a LEFT JOIN qy_coin_data b ON a.id = b.coin_id LEFT JOIN qy_coin_tags c ON a.id = c.coin_id
              WHERE b.day = '{}' 
          '''.format(statistic_date)

    df_coin_standard = pd.read_sql(sql, engine)

    df_coin_standard = df_coin_standard[df_coin_standard['tags'] != '稳定币']
    market_value = df_coin_standard.groupby('cate').sum()
    token_market_value = market_value.loc[2, 'market_cap']  # tokens
    coin_market_value = market_value.loc[1, 'market_cap']  # coins
    token_number = int(round(10 * token_market_value / (token_market_value + coin_market_value)))
    coin_number = int(round(10 * coin_market_value / (token_market_value + coin_market_value)))

    # 缩减至30只成分币
    df_coin_standard.sort_values(by='market_cap', ascending=False, inplace=True)
    df_coin_standard.drop_duplicates(subset='id', keep='first', inplace=True)
    # coin_ids = df_coin_standard['smyt_id'].drop_duplicates().tolist()
    token_set = set(df_coin_standard[df_coin_standard['cate'] == 2]['id'].tolist())
    coin_set = set(df_coin_standard[df_coin_standard['cate'] == 1]['id'].tolist())
    token_sample = token_set & component_stable & component_rank
    coin_sample = coin_set & component_stable & component_rank


    rank_token_total = []
    rank_coin_total = []

    date_zip = zip(first_date, last_date)

    for date in date_zip:
        sql = "select coin_id,market_cap,volume from qy_coin_data where day BETWEEN '{}' and '{}' order by day" \
            .format(date[0], date[1])
        df = pd.read_sql(sql, engine)
        df_token = df[df['coin_id'].isin(token_sample)]
        df_coin = df[df['coin_id'].isin(coin_sample)]
        token_market_value_index = df_token.groupby('coin_id').sum().sort_values(by='market_cap',
                                                                                 ascending=False).index.drop_duplicates().tolist()
        token_turnover_index = df_token.groupby('coin_id').sum().sort_values(by='volume',
                                                                             ascending=False).index.drop_duplicates().tolist()
        coin_market_value_index = df_coin.groupby('coin_id').sum().sort_values(by='market_cap',
                                                                               ascending=False).index.drop_duplicates().tolist()
        coin_turnover_index = df_coin.groupby('coin_id').sum().sort_values(by='volume',
                                                                           ascending=False).index.drop_duplicates().tolist()
        rank_token = {}
        rank_coin = {}
        for coin_id in token_sample:
            try:
                rank_token[coin_id] = token_market_value_index.index(coin_id) + token_turnover_index.index(coin_id)
            except:
                print('data missing: %s' % (coin_id))
        for coin_id in coin_sample:
            try:
                rank_coin[coin_id] = coin_market_value_index.index(coin_id) + coin_turnover_index.index(coin_id)
            except:
                print('data missing: %s' % (coin_id))

        rank_token_total.append(rank_token)
        rank_coin_total.append(rank_coin)
    rank_token_total = pd.DataFrame(rank_token_total).mean()
    rank_coin_total = pd.DataFrame(rank_coin_total).mean()
    rank_token_total = rank_token_total.sort_values(ascending=True).index.tolist()
    rank_coin_total = rank_coin_total.sort_values(ascending=True).index.tolist()

    if len(rank_token_total) < token_number:
        component_token = set(rank_token_total)
        component_coin = set(rank_coin_total[:coin_number + token_number - len(rank_token_total)])
    else:
        if len(rank_coin_total) < coin_number:
            component_coin = set(rank_coin_total)
            component_token = set(rank_token_total[:token_number + coin_number - len(rank_coin_total)])
        else:
            component_token = set(rank_token_total[:token_number])
            component_coin = set(rank_coin_total[:coin_number])
    component_coin = component_token | component_coin

    return component_coin


@numba.jit
def index_statistics(market_value_list):
    """
    成分虚拟币的市值权重
    :param market_value_list:
    :return df_statistics:
    """
    df_statistics = pd.DataFrame(market_value_list, columns=['coin_id', 'market_value', 'ratio'])
    df_statistics.index = df_statistics.coin_id
    del df_statistics['coin_id']

    df_statistics['adjust_market_value'] = df_statistics.market_value * df_statistics.ratio
    a = 0.25
    while max(df_statistics.adjust_market_value) / sum(df_statistics.adjust_market_value) > a:
        # df_statistics.sort_values(by='adjust_market_value', ascending=False, inplace=True)
        for coin_id in df_statistics.index.tolist():
            weight = (df_statistics.loc[coin_id, 'adjust_market_value'] / sum(df_statistics.adjust_market_value))
            if weight > a:
                df_statistics.loc[coin_id, 'ratio'] = df_statistics.loc[coin_id, 'ratio'] - 0.001
                # df_statistics.loc[coin_id, 'ratio'] =  round(a*(df_statistics.adjust_market_value.sum() - \
                #                                      df_statistics.loc[coin_id,'adjust_market_value'])/\
                #                                   (1-a)/df_statistics.loc[coin_id,'market_value'],4)

        df_statistics.adjust_market_value = df_statistics.market_value * df_statistics.ratio

    return df_statistics


def all_coin_index():
    """
    虚拟币指数
    :param:
    :return indicators:
    """
    indicators = []
    sql = "select coin_id, day, market_cap from qy_coin_data where day BETWEEN '{}' and '{}' order by day" \
        .format(dt.datetime.strptime('2017-01-01', '%Y-%m-%d'), dt.datetime.today())
    df = pd.read_sql(sql, engine)
    max_day = df['day'].max()
    df.fillna(method='ffill', inplace=True)

    today = dt.datetime.today().strftime('%Y-%m-%d')
    if today[8:] == '01':
        try:
            component_dict = list(db.coin_component.find({}, {'_id': 0}))[0]
            weight_dict = list(db.coin_weight.find({}, {'_id': 0}))[0]
            if list(component_dict.keys())[-1] != today:
                component_all()
        except:
            component_all()
            component_dict = list(db.coin_component.find({}, {'_id': 0}))[0]
            weight_dict = list(db.coin_weight.find({}, {'_id': 0}))[0]
    else:
        try:
            component_dict = list(db.coin_component.find({}, {'_id': 0}))[0]
            weight_dict = list(db.coin_weight.find({}, {'_id': 0}))[0]
        except:
            component_all()
            component_dict = list(db.coin_component.find({}, {'_id': 0}))[0]
            weight_dict = list(db.coin_weight.find({}, {'_id': 0}))[0]

    # 基期市值
    list_base = []
    component = component_dict['2017-01-01']
    weight = weight_dict['2017-01-01']
    component = [i for i in component if ~np.isnan(i) ]
    weight = [i for i in weight if 0 <= i <= 1]

    # for coin_id in component:
    #     market_value = df[df.coin_id == coin_id].iloc[0].market_cap
    #     list_base.append([coin_id, market_value, 1.])

    #market_value_base = sum(index_statistics(list_base).adjust_market_value)
    for j in range(len(component)):
        df_coin = df[df.coin_id == component[j]]
        df_coin.index = pd.to_datetime(df_coin.day)
        list_base.append(df_coin.loc['2017-01-01', 'market_cap'] * weight[j])
    #    list_daily.append([coin_id, market_value, 1.])

    market_value_base = sum(list_base)
    # 每日市值
    market_date_list = list(pd.date_range(start='2017-1-1', end=max_day))
    for market_date in market_date_list:
        statistic_date = market_date.strftime("%Y-%m-%d")
        component_date = statistic_date[0:8] + '01'
        component = component_dict[component_date]
        weight = weight_dict[component_date]
        component = [i for i in component if ~np.isnan(i)]
        weight = [i for i in weight if 0 <= i <= 1]

        # list_daily = []
        adjust_market_value = []
        for j in range(len(component)):
            df_coin = df[df.coin_id == component[j]]
            df_coin.index = pd.to_datetime(df_coin.day)
            df_coin = df_coin.reindex(index=market_date_list, method='ffill')
            adjust_market_value.append(df_coin.loc[statistic_date, 'market_cap'] * weight[j])
        #    list_daily.append([coin_id, market_value, 1.])

        market_value_daily = sum(adjust_market_value)

        # 虚拟币指数

        coin_index_value = round(market_value_daily * 100 / market_value_base, 4)
        # 基期基数调整
        if (market_date + dt.timedelta(1)).day == 1:
            nextdate = (dt.datetime.strptime(statistic_date, "%Y-%m-%d") + relativedelta(days=1)).strftime('%Y-%m-%d')
            component = component_dict[nextdate]
            weight = weight_dict[nextdate]
            component = [i for i in component if ~np.isnan(i) ]
            weight = [i for i in weight if 0 <= i <= 1]

            adjust_market_value_base = []
            for j in range(len(component)):
                df_coin = df[df.coin_id == component[j]]
                df_coin.index = pd.to_datetime(df_coin.day)
                df_coin = df_coin.reindex(index=market_date_list, method='ffill')
                adjust_market_value_base.append(df_coin.loc[statistic_date, 'market_cap'] * weight[j])

            market_value_new = sum(adjust_market_value_base)
            market_value_base = market_value_base * market_value_new / market_value_daily

        indicators.append({"statistic_date": statistic_date, 'value': coin_index_value})
        pprint(indicators[-1])

    return indicators

#
# # 指数回测，以BTC为基准
# def back_test(indicators):
#     sql = "SELECT statistic_date,current_price_usd as price FROM coin_trading_copy WHERE coin_id = 'JRB100001' and statistic_date between '{}' and '{}' order by statistic_date". \
#         format(dt.datetime.strptime(indicators['statistic_date'][0], '%Y-%m-%d'),
#                dt.datetime.strptime(indicators['statistic_date'][len(indicators) - 1], '%Y-%m-%d'))
#     btc = pd.read_sql(sql, engine)
#     hab_index = indicators
#     btc.index = btc.statistic_date
#     btc = btc.reindex(index=pd.to_datetime(hab_index['statistic_date']).tolist(), method='ffill')
#     btc['profit'] = btc['price'] / btc.iloc[0, 1] - 1
#     hab_index['profit'] = hab_index['value'] / hab_index.loc[0, 'value'] - 1
#     # 绘制收益曲线
#     figuresize = 11, 9
#     plt.figure(figsize=figuresize)
#     plt.title('HAB_index acc_profit')
#     plt.plot(btc['statistic_date'], hab_index['profit'], color='red', label='HAB30')
#     plt.plot(btc['statistic_date'], btc['profit'], color='blue', label='BTC')
#     plt.legend()
#     plt.tick_params(labelsize=12)
#     plt.ylabel('Acc_profit')
#     plt.savefig('d:/habo/data/HAB30/HAB_BTC_a0.25.png')
#     plt.show()
#
#     hab = hab_index['value'].tolist()
#     # 夏普率
#     # 年化收益
#     index_profit_a = (hab[-1] / hab[0]) ** (365 / len(hab)) - 1
#     # 标准差
#     index_profit = (hab_index['value'] / hab_index['value'].shift(1) - 1).tolist()
#     index_profit[0] = 0
#     index_profit_avg = np.mean(index_profit)
#     index_std_a = (sum([(index_profit[i] - index_profit_avg) ** 2 for i in range(len(index_profit))]) / (
#             len(index_profit) - 1)) ** 0.5 * 365 ** 0.5
#     sharpe = index_profit_a / index_std_a
#     print('年化收益：', index_profit_a)
#     print('夏普率：', sharpe)
#
#     # 最大回撤
#     backtrace = []
#     for i in range(len(hab)):
#         temp = hab[i] / max(hab[:i + 1]) - 1
#         if temp > 0:
#             temp = 0
#         backtrace.append(temp)
#     MBT = min(backtrace)
#     print('最大回撤：', MBT)
#
#     # 累计收益
#     hab_acc_profit = hab[-1] / hab[0] - 1
#     print('HAB累计收益', hab_acc_profit)
#     # 基准收益
#     btc_acc_profit = btc['profit'][len(btc) - 1]
#     print('BTC累计收益：', btc_acc_profit)


def component_weight(date):
    date = dt.datetime.strptime(date, '%Y-%m-%d')
    if int((date.month - 1) / 3) * 3 + 1 < 10:
        component_date = str(date.year) + '-0' + str(int((date.month - 1) / 3) * 3 + 1) + '-01'
    else:
        component_date = str(date.year) + '-' + str(int((date.month - 1) / 3) * 3 + 1) + '-01'

    component = pd.read_sql('select * from qy_coin_component where date = "{}"'.format(component_date),engine)

    component_n = component['coin_id'].tolist()
    ratio_n = component['ratio'].tolist()

    weight = []
    sql = '''SELECT a.coin_id,a.day,a.market_cap FROM qy_coin_data a,
    	(SELECT coin_id,MAX(DAY) AS DATE FROM qy_coin_data WHERE DAY <= '{}' GROUP BY coin_id) b 
     WHERE a.coin_id = b.coin_id AND a.day = b.date'''.format(date)
    df = pd.read_sql(sql, engine)
    df.set_index(['coin_id'], 1, inplace=True)
    for i in range(len(component_n)):
        weight.append(df.loc[component_n[i]]['market_cap'] * ratio_n[i])

    weight = pd.DataFrame(weight, index=component_n, columns=['adjust_market_value'])
    weight['weight'] = weight['adjust_market_value'] / weight['adjust_market_value'].sum()
    weight.fillna(0, inplace=True)
    del weight['adjust_market_value']
    return weight


if __name__ == '__main__':
     component_all()
     all_coin_index()