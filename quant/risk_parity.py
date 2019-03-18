####### step1
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize
from database import engine_test
from sklearn import linear_model
import datetime as dt

# 读取收盘价数据
def read_data():
    coin_ids = (1, 3, 4, 7, 9, 11, 12, 1341, 1347, 1360)
    df = pd.read_sql(
        'select a.coin_id,a.day,a.close_p,a.market_cap, b.symbol from qy_coin_data a left join qy_coin_info b on a.coin_id = b.id where a.day >= "2017-01-01" and a.coin_id in {} order by a.coin_id,a.day'.format(
            coin_ids),
        engine_test())
    coin_ids = df.drop_duplicates('coin_id').coin_id.tolist()
    coin_ids = np.array(coin_ids).astype(str)
    date_range = pd.date_range('2017-01-01', '2018-12-16', freq='D')
    df_1 = pd.DataFrame(columns=coin_ids, index=date_range)

    for id in coin_ids:
        if id == 1: continue
        df_id = df[df.coin_id == int(id)]
        df_id.index = pd.to_datetime(df_id.day)
        df_id = df_id.reindex(index=date_range, method='ffill')
        # if df_id['close_p'].isna().any():
        #     del df_1[id]
        #     continue
        df_id['coin_id'] = int(id)
        df_1[id] = df_id.close_p

    return df_1


# 计算对数收益率
def calculate_log_return(data):
    coin_ids = data.columns.tolist()

    for j in coin_ids:
        data['r_' + str(j)] = np.log(data[j] / data[j].shift(1))

    return data.iloc[1::, 0::]


# 将收益率进行标准化处理，形成新的表格std_data
def standardlize_process(data):
    scaler = StandardScaler()
    scaler.fit(data)
    std_data = pd.DataFrame(scaler.transform(data), columns=data.columns, index=data.index)
    std_data = std_data.iloc[:, ::]
    return std_data


# 资产收益率序列相关性检验：滚动计算资产的协方差矩阵
def cov_asset(std_data):
    return std_data.rolling(window=30).cov().iloc[len(std_data.columns)*29::, 0::]
    # 风险度量期间为一年，故滚动窗口设置为12个月，从2012-12月末到2018-02月末共计 63个月，每个月6条记录，共63*6=378条记录

##### step2

# 计算每类资产对总资产组合的风险贡献
def calculate_risk_contribution(weight, cov_matrix):
    sigma = np.sqrt(weight * cov_matrix * np.matrix(weight).T)
    # 边际风险贡献 Marginal Risk Contribution (MRC)
    MRC = cov_matrix * np.matrix(weight).T / sigma
    # 风险贡献 Risk Contribution (RC)
    RC = np.multiply(MRC, np.matrix(weight).T)
    return RC


# 定义优化问题的目标函数，即最小化资产之间的风险贡献差
def risk_budget_objective(x_weight, parameters):
    # x_weight是带求解的各个大类资产下面子标的的权重,parameters是下面函数中的args=[one_cov_matrix,RC_set_ratio]参数传递
    # x_weight的初始值是函数 calculate_portfolio_weight中的 weight0

    # 协方差矩阵,也即函数 calculate_portfolio_weight 中传递的 args中的第一个参数 one_cov_matrix
    one_cov_matrix = parameters[0]
    # 风险平价下的目标风险贡献度向量，也即函数 calculate_portfolio_weight 中传递的 args中的第二个参数 RC_set_ratio
    N = parameters[1]
    # RC_target为风险平价下的目标风险贡献，一旦参数传递以后，RC_target就是一个常数，不随迭代而改变
    RC_target = np.sqrt(x_weight * one_cov_matrix * np.matrix(x_weight).T) / N
    # RC_real是 每次迭代以后最新的真实风险贡献，随迭代而改变
    RC_real = calculate_risk_contribution(x_weight, one_cov_matrix)
    sum_squared_error = sum(np.square(RC_real - RC_target.T))[0,0]
    return sum_squared_error


# 优化问题的第一个约束条件
def constraint1(x_weight):
    return np.sum(x_weight) - 1.0


# 优化问题的第二个约束条件
def constraint2(x_weight):
    return x_weight


# 根据资产预期目标风险贡献度来计算各资产的权重
def calculate_portfolio_weight(N, one_cov_matrix):
    weight0 = [1/N]*N
    cons = ({'type': 'eq', 'fun': constraint1}, {'type': 'ineq', 'fun': constraint2})
    res = minimize(risk_budget_objective, weight0, args=[one_cov_matrix, N], method='SLSQP',
                   constraints=cons, options={'disp': True})
    weight_final = np.asmatrix(res.x)
    return weight_final


## 计算风险平价下每月的资产配置权重
# 假设1：四个资产的风险贡献度相等
# 假设2：同类资产下每个标的的风险贡献相同



#### step3：回测Backtest
## 基于过去一年的风险分配权重，计算各资产的月度收益率
# def calculate_RC_monthly_return(file):
#     std_df = standardlize_process(file)
#     date_list = []
#     for i in range(63):
#         date_list.append(str(2012 + int((i + 11) / 12)) + '-' + month[(i - 1) % 12])
#     # date_list
#     _std_df = std_df.iloc[11::, 0::].T
#     _std_df.columns = date_list
#     __std__df = _std_df.T
#     __std__df['key'] = __std__df.index
#     df_monthly_weight['key'] = df_monthly_weight.index
#     # 合并表，将月收益数据和权重表横向合并
#     monthly_return = pd.merge(__std__df, df_monthly_weight, on='key')
#     monthly_return.set_index('key', inplace=True)
#
#     column_list = ['return_IF00.CFE', 'return_IC00.CFE', 'return_TF00.CFE', 'return_AU00.SHF', 'return_159920.OF',
#                    'return_511880.SH', 'IF00.CFE', 'IC00.CFE', 'TF00.CFE', 'AU00.SHF', '159920.OF', '511880.SH']
#     # 计算加权后的各资产收益率：权重乘以资产收益率
#     for i in range(6):
#         monthly_return['weight' + '_' + column_list[i]] = monthly_return[column_list[i]] * monthly_return[
#             column_list[i + 6]]
#     monthly__return = monthly_return.iloc[0::, -6::]
#     monthly__return['all_portfolio_return'] = monthly__return.apply(lambda x: x.sum(), axis=1)
#     # 用债券收益率作为benchmark
#     monthly__return['benchmark'] = monthly_return['return_TF00.CFE']
#     return monthly__return


## 收益回测，模型评价
# def backtest_model1(monthly_data):
#     # 设置评价指标
#     total_return = {}
#     annual_return = {}
#     excess_return = {}
#     annual_volatility = {}
#     sharpe = {}
#     information_ratio = {}
#     win_prob = {}
#     drawdown = {}
#     tr = {}
#
#     # 单独计算benchmark的相关指标
#     portfolio_list = ['weight_return_IF00.CFE', 'weight_return_IC00.CFE', 'weight_return_TF00.CFE',
#                       'weight_return_AU00.SHF', 'weight_return_159920.OF', 'weight_return_511880.SH',
#                       'all_portfolio_return']
#     bench_total_return = (monthly_data['benchmark'] + 1).T.cumprod()[-1] - 1
#     bench_annual_return = (float(bench_total_return) + 1.0) ** (1. / (5 + 1 / 6)) - 1
#
#     # 每一种指标的具体构建方法
#     for i in portfolio_list:
#         monthly = monthly_data[i]
#         total_return[i] = (monthly + 1).T.cumprod()[-1] - 1
#         annual_return[i] = (float(total_return[i]) + 1.0) ** (1. / (5 + 1 / 6)) - 1
#         annual_volatility[i] = monthly.std()
#         sharpe[i] = (annual_return[i] - bench_annual_return) / annual_volatility[i]
#         drawdown[i] = monthly.min()
#         win_excess = monthly - monthly_data['benchmark']
#         win_prob[i] = win_excess[win_excess > 0].count() / float(len(win_excess))
#
#     # 将字典转换为dataframe
#     ar = pd.DataFrame(annual_return, index=monthly.index).drop_duplicates().T
#     tr = pd.DataFrame(total_return, index=monthly.index).drop_duplicates().T
#     av = pd.DataFrame(annual_volatility, index=monthly.index).drop_duplicates().T
#     sp = pd.DataFrame(sharpe, index=monthly.index).drop_duplicates().T
#     dd = pd.DataFrame(drawdown, index=monthly.index).drop_duplicates().T
#     wp = pd.DataFrame(win_prob, index=monthly.index).drop_duplicates().T
#     ar['key'] = ar.index  # 年化收益
#     tr['key'] = tr.index  # 累积收益
#     av['key'] = av.index  # 年化波动
#     sp['key'] = sp.index  # 夏普比率
#     dd['key'] = dd.index  # 最大回撤
#     wp['key'] = wp.index  # 胜率
#     backtest_df = pd.merge(ar, pd.merge(tr, pd.merge(av, pd.merge(sp, pd.merge(dd, wp, on='key'), on='key'), on='key'),
#                                         on='key'), on='key')
#     backtest_df.set_index('key', inplace=True)
#     backtest_df.columns = ['annual return', 'total return', 'annual volatility', 'sharpe ratio', 'drawdown', 'win prob']
#     return backtest_df
#计算RSRS指标
def RSRS(N,M,coin_ids):
    sql = '''select * from qy_coin_data where coin_id in {} order by day'''.format(coin_ids)
    df = pd.read_sql(sql, engine_test())
    df['day'] = pd.to_datetime(df['day'])
    regr = linear_model.LinearRegression()
    df_new = pd.DataFrame(columns=['coin_id','gold', 'dead'])
    for id in coin_ids:
        df_coin = df[df['coin_id'] == id]
        if len(df_coin) < M:
            print("don't have enough data")
            return False
        date_list = pd.date_range(df_coin.iloc[0]['day'], (dt.date.today() - dt.timedelta(1)).strftime('%Y-%m-%d'))
        df_coin = df_coin.copy()
        df_coin.index = df_coin['day']
        del df_coin['day']
        df_coin.sort_index(ascending=True, inplace=True)
        df_coin = df_coin.reindex(index=date_list, method='ffill')
        df_coin.fillna(0, inplace=True)
        df_coin.coin_id = id
        coef_N = []
        #R2 = []

        for i in range(len(df_coin)):
            if i < N-1:
                #regr.fit(np.array(df_coin.iloc[0:i+1]['low']).reshape(-1,1), np.array(df_coin.iloc[0:i+1]['high']).reshape(-1,1))
                coef_N.append(0)
                #R2.append(0)
            else:
                regr.fit(np.array(df_coin.iloc[i-N+1:i + 1]['low_p']).reshape(-1, 1),
                                np.array(df_coin.iloc[i-N+1:i + 1]['high_p']).reshape(-1, 1))
                coef_N.append(regr.coef_[0][0])
                # R2.append(regr.score(np.array(df_coin.iloc[i-N+1:i + 1]['low']).reshape(-1, 1),
                #                 np.array(df_coin.iloc[i-N+1:i + 1]['high']).reshape(-1, 1)))
        df_coin['coef_N'] = coef_N
        df_coin['avg'] = df_coin['coef_N'].rolling(M).mean()
        df_coin['std'] = df_coin['coef_N'].rolling(M).std()
        df_coin['RSRS'] = (df_coin['coef_N'] - df_coin['avg']) / df_coin['std']
        #df_coin['RSRS_R2'] = df_coin['RSRS']
        df_coin['gold'] = (df_coin['RSRS'] > 1)
        df_coin['dead'] = (df_coin['RSRS'] < 0.1)
    #df_coin.to_csv('d://RSRS_bitcoin.csv')
        df_coin = df_coin[['coin_id', 'gold', 'dead']]
        df_new = pd.concat([df_new, df_coin], join='outer')
    return df_new

def benchmark():
    sql = "select * from qy_coin_data where coin_id = 1 and day >= '2017-01-01'"
    date_range = pd.date_range('2017-01-30', '2018-12-16', freq='D')
    df = pd.read_sql(sql,engine_test())
    df.index = pd.to_datetime(df['day'])
    df = df.reindex(index=date_range,method='ffill')
    df['new'] = df['close_p'] / df.iloc[0]['close_p']
    price = df['new'].tolist()
    #print(df['day'])

    return price

def main():
    cash = 1000000
    cost = 0.002
    wealth = []
    K = 10
    #N = 10
    data = calculate_log_return(read_data())
    coin_ids = (1, 3, 4, 7, 9, 11, 12, 1341, 1347, 1360)
    #RSRS指标
    df_RSRS = RSRS(14, 110, coin_ids)
    coin_pool = {}
    coin_ids = np.array([1, 3, 4, 7, 9, 11, 12, 1341, 1347, 1360]).astype(str)
    #print(data)
    for j in range(len(coin_ids)):
        coin_pool.update({coin_ids[j]: [0, 0]})

    for i in range(360, len(data.index)):
        temp = 0
        #获取每天的RSRS指标，有可能为空
        df_RSRS_day = df_RSRS.loc[data.index[i]]
        for key in coin_pool:
            if coin_pool[key][0] > 0:
                coin_pool[key][1] = coin_pool[key][0] * data.iloc[i][key]
                temp += coin_pool[key][1]
        marketcap = temp
        if data.index[i].strftime('%Y-%m-%d')[-2:] == '01' or i == 360:
            cash_t = 0

            for key in coin_pool:
                cash_t += coin_pool[key][1] * (1 - cost)
                coin_pool[key] = [0, 0]
            cash = cash + cash_t
            marketcap = 0
            df = data.iloc[i, int(len(data.columns)/2):]
            df_date = data.iloc[i-29:i+1][df.index.tolist()]
            df_date.dropna(axis=1, inplace=True)
            N = len(df_date.columns)
            std_data = standardlize_process(df_date)
            #std_data = df_date
            cov_matrix = std_data.cov().values
            cov_matrix = np.diag(np.diag(cov_matrix))
            weight_final = calculate_portfolio_weight(N,cov_matrix)
            print(weight_final)
            coin_ids = df_date.columns.tolist()

            for j in range(len(coin_ids)):
                df_RSRS_coin = df_RSRS_day[df_RSRS_day['coin_id']==int(coin_ids[j][2:])]
                if ~df_RSRS_coin.empty:
                    if df_RSRS_coin.iloc[0]['dead']:
                        weight_final[0, j] = 0
                    # elif df_RSRS_coin.iloc[0]['gold']:
                    #     weight_final[0, j] = weight_final[0, j] * K
            weight_final = weight_final / weight_final.sum()

            for j in range(len(coin_ids)):
                coin_pool[coin_ids[j][2:]][1] = weight_final[0, j] * cash
                coin_pool[coin_ids[j][2:]][0] = coin_pool[coin_ids[j][2:]][1] / data.iloc[i][coin_ids[j][2:]]
            marketcap = cash
            cash = 0
        # else:
        #     for key in coin_ids:
        #         df_RSRS_coin = df_RSRS_day[df_RSRS_day['coin_id']==int(key[2:])]
        #         if df_RSRS_day.empty: continue
        #         #if coin_pool[key][0] > 0:
        #         if df_RSRS_coin.iloc[0]['dead'] and coin_pool[key[2:]][0] > 0:
        #             cash = cash + coin_pool[key[2:]][0] * data.iloc[i][key[2:]] * (1 - cost)
        #             marketcap = marketcap - coin_pool[key[2:]][0] * data.iloc[i][key[2:]]
        #             coin_pool[key[2:]][0] = 0
        #             coin_pool[key[2:]][1] = 0
        #         elif df_RSRS_coin.iloc[0]['gold'] and cash > 0:
        #             coin_pool[key[2:]][1] = coin_pool[key[2:]][1] + cash / 2
        #             coin_pool[key[2:]][0] = coin_pool[key[2:]][1] / data.iloc[i][key[2:]]
        #             marketcap = marketcap + cash / 2
        #             cash = cash - cash / 2
            #print(weight_final)
            # print(np.shape(weight_final))
            # print(coin_ids)
            # print(i)
            # print(weight_final[0,1])
        #
        #     print('第%d个月金额%d' % (i,cash))
        wealth.append((cash + marketcap)/1000000)
        print(wealth[-1])
    #     cash = 0
    #
    # bench = benchmark()
    # # print(len(bench))
    # # print(len(wealth))
    # plt.plot(wealth)
    # plt.plot(bench,color='r')
    # plt.show()


if __name__ == '__main__':
    main()