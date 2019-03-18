import pandas as pd
import numpy as np
from scipy import optimize
from database import engine_test
from statsmodels.tsa.stattools import adfuller
from adf_test import *
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA
import statsmodels.api as sm
from statsmodels.graphics.api import qqplot
import itertools
import statsmodels.tsa.stattools as sts
from statsmodels.stats import diagnostic
from scipy.stats import normaltest
from adf_test import *
from statsmodels.stats.diagnostic import acorr_ljungbox

#读取数据
def read_data(coin_id):
    sql = 'select coin_id,day,close_p from qy_coin_data where coin_id = {} and close_p > 0 and day >= "2016-01-01" and day < "2018-10-10" order by day'.format(coin_id)
    df = pd.read_sql(sql, engine_test())
    if len(df) < 1:
        return df
    date_range = pd.date_range(df.iloc[0]['day'], df.iloc[-1]['day'],freq='D')
    df.index = pd.to_datetime(df['day'])
    df = df.reindex(index=date_range,method='ffill')
    df.dropna(inplace=True)
    df = df['close_p']
    df = np.log(df)
    df = df.diff()
    df.dropna(inplace=True)
    #df['return'] = df['close_p'] / df['close_p'].shift(1) - 1
    #df.dropna(inplace=True)
    #df = df['return']
    return df


def decompose(ts_log, model='additive'):
    decomposition = seasonal_decompose(ts_log, model)
    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid
    trend.plot(color='red',label='trend')
    #seasonal.plot(color='blue',label='seasonal')
    #residual.plot(color='yellow',label='residual')
    plt.show()

def params_select(df):
    resid = sm.tsa.arma_order_select_ic(df,max_ar=4,max_ma=4,ic='aic')
    print('AIC:{}'.format(resid.aic_min_order))
    #print('BIC:{}'.format(resid.bic_min_order))
    #print('HQIC:{}'.format(resid.hqic_min_order))


if __name__ == '__main__':
    coin = pd.read_sql('select id from qy_coin_info where from_unixtime(updatetimestamp) > date_sub(now(),interval 7 day)',engine_test())
    coin_ids = coin['id'].tolist()
    data_unstable = []
    data_random = []
    data_target = []
    coin_ids = [7]

    for coin_id in coin_ids:
        df = read_data(coin_id)
        if len(df) < 500:
            continue
        adf = testStationarity(df)
        rand_test = acorr_ljungbox(df)[1][0]
        if adf['p-value'] > 0.05:
            data_unstable.append(coin_id)
        else:
            if rand_test < 0.05:
                data_target.append(coin_id)
                print(coin_id)
                #params_select(df)
            else:
                data_random.append(coin_id)

    print('可建模的币数量',len(data_target))
    #平稳性检验
    print(testStationarity(df))
    #acf,pacf图
    draw_acf_pacf(df)

    #正态分布检验--------------
    # plt.hist(df)
    # plt.show()
    # print(normaltest(df))
    #---------------------------
    #decompose(df)
    #df =df.diff().dropna()
    #params_select(df)
    df_train = df['2016':'2017']
    df_test = df['2018-01-01':'2018-10-01']
    history = df_train.tolist()
    #残差自相关性检验
    # model = ARIMA(history, order=(0, 1, 1)).fit(disp=0)
    # resid = model.resid
    # print(sm.stats.durbin_watson(resid))
    predictions = []
    cash = 1000000
    cost = 0.002
    num = 0
    wealth = []

    for i in range(len(df_test)):
        try:
            model = sm.tsa.ARMA(history,(3, 3)).fit(disp=0)
            #print(model.summary())
            output = model.forecast()
            predictions.append(output[0][0])
            print(output[0][0],df_test[i])
            history.append(df_test.iloc[i])
            #history.append(output[0][0])
            if i == 0:
                if predictions[-1] > df_train[-1]:
                    num = cash / np.exp(df_train[-1])
                    wealth.append(cash / 1000000)
                    cash = 0
            #        print('fisrt buy',num,cash)
            else:
                if predictions[-1] > df_test[i-1] and cash > 0:
                    num = cash / np.exp(df_test[i-1])
                    wealth.append(cash / 1000000)
                    cash = 0
            #        print('buy', num, cash)
                elif predictions[-1] < df_test[i-1] and num > 0:
                    cash = num * np.exp(df_test[i-1]) * (1 - cost)
                    num = 0
                    wealth.append(cash / 1000000)
            #        print('sell', num, cash)
        except Exception as e:
            print(history[-1])
            print(str(e))

    predictions = pd.Series(predictions,index=df_test.index)
    print(pd.DataFrame(wealth))
    plt.plot(wealth)
    plt.show()
    return_predict = pd.Series(predictions,index=df_test.index)
    return_test = df_test
    return_predict = predictions.diff().dropna()
    return_test = df_test.diff().dropna()
    #模型准确率评价-------------------------------------------------
    flag_up = 0
    flag_down = 0
    for i in range(1,len(return_predict)):
        if return_predict[i] > 0 and return_test[i] > 0:
            flag_up += 1
        if return_predict[i] < 0 and return_test[i] < 0:
            flag_down += 1

    print(flag_up,flag_down,len(return_predict),len(return_test))
    #--------------------------------------------------------------------
    plt.plot(predictions,color='r')
    plt.plot(df_test)
    plt.show()
#另一种方法-------------------------------------------------------------
    # model = sm.tsa.statespace.SARIMAX(df,order=(0,1,1),seasonal_order=()).fit(disp=0)
    # print(model.summary().tables[1])
    # model.plot_diagnostics()
    # plt.show()
    # pred = model.get_prediction(start=pd.to_datetime('2018-01-01'),dynamic=False)
    # pred_ci = pred.conf_int()
    # print(pred.predicted_mean)
    # ax = df['2017':].plot(label='origin')
    # pred.predicted_mean.plot(ax=ax,label='predict')
    # ax.fill_between(pred_ci.index,
    #                 pred_ci.iloc[:, 0],
    #                 pred_ci.iloc[:, 1], color='k', alpha=.2)
    #
    # ax.set_xlabel('Date')
    # ax.set_ylabel('price')
    # plt.legend()
    #
    # plt.show()
    #------------------------------------------------------------------------
    # predict = pred.predicted_mean;
    # origin = df['2018-01-01':]
    # MSE = ((predict - origin) ** 2).mean()
    #print('MSE:',MSE)
    # print(model.summary())
    # residuals = pd.DataFrame(model.resid)
    # residuals.plot()
    # plt.show()
    # residuals.plot(kind='kde')
    # plt.show()
    #result_arma = model.fit(disp=-1, method='css')
    #检验残差自相关性
    # print(sm.stats.durbin_watson(model1.resid.values))
    # resid = model1.resid
    # fig = plt.figure(figsize=(12, 8))
    # ax = fig.add_subplot(111)
    # fig = qqplot(resid, line='q', ax=ax, fit=True)
    # plt.show()



