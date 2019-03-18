import pandas as pd
import numpy as np
from scipy import optimize
from database import engine_test
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats import diagnostic
import statsmodels.tsa.stattools as sts
from adf_test import *
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA
import statsmodels.api as sm
from statsmodels.graphics.api import qqplot
import itertools
from arch import arch_model
from arch.univariate import ZeroMean, GARCH, StudentsT, ConstantMean
from statsmodels.stats.diagnostic import acorr_ljungbox
import datetime as dt


#读取数据
def read_data():
    sql = 'select * from qy_coin_data where coin_id = 7 and day >= "2016-01-01" and day < "2018-10-10" order by day'
    df = pd.read_sql(sql, engine_test())
    date_range = pd.date_range('2016-01-01','2018-10-09',freq='D')
    df.index = pd.to_datetime(df['day'])
    df = df.reindex(index=date_range,method='ffill')
    df.dropna(inplace=True)
    df = df['close_p']
    df = np.log(df)

    return df

#收益率平稳性检验
def adftest(df):
    _,p_value,*_ = adfuller(df)
    if p_value < 0.05:
        print('收益率平稳',p_value)
    else:
        print('收益率不平稳',p_value)

def params_select(df):
    resid = sm.tsa.arma_order_select_ic(df,max_ar=4,max_ma=4,ic=['aic','bic','hqic'])
    print('AIC:{}'.format(resid.aic_min_order))
    print('BIC:{}'.format(resid.bic_min_order))
    print('HQIC:{}'.format(resid.hqic_min_order))

def SARIMA(df):
    model = sm.tsa.statespace.SARIMAX(df,order=(0,1,1)).fit(disp=0)
    print(model.summary().tables[1])
    model.plot_diagnostics()
    plt.show()
    #pred = model.get_prediction(start=pd.to_datetime('2018-01-01'),dynamic=False)
    pred = model.forecast(steps=5)
    pred_ci = pred.conf_int()
    print(pred.predicted_mean)

    ax = df['2017'].plot(label='origin')
    pred.predicted_mean.plot(ax=ax,label='predict')
    ax.fill_between(pred_ci.index,
                    pred_ci.iloc[:, 0],
                    pred_ci.iloc[:, 1], color='k', alpha=.2)

    ax.set_xlabel('Date')
    ax.set_ylabel('price')
    plt.legend()

    plt.show()
# ---------------

if __name__ == '__main__':
    df = read_data()
    #SARIMA(df['2017'])
    df1 = df.diff().dropna()
    # df1 =df.diff(1)
    # df1.dropna(inplace=True)
    df2 = df1["2016":"2017"]
    #收益率平稳性检验结果为1阶弱平稳
    #adftest(df2)
    #参数选择为(0,1)
    params_select(df1)
    print(acorr_ljungbox(df1, lags=1))
    #系数估计------------------------------------------------
    model = sm.tsa.ARMA(df2,(0,1)).fit()
    print(model.summary())
    print('----------------------------------------')
    print(model.params)
    #--------------------------------------------------------
    #收益率残差自相关性检验-----------------------------------
    resid = model.resid
    print(sm.stats.durbin_watson(resid.values))
    #检验残差arch效应-----------------------------------------
    *_, fpvalue = diagnostic.het_arch(resid)
    if fpvalue < 0.05:
        print('异方差性显著', fpvalue)
    else:
        print('异方差性不显著', fpvalue)
    #建立arch模型-----------------------------------------------
    #模型预测
    model = sm.tsa.ARMA(df2, (0, 1)).fit()
    arch_mod = ConstantMean(df2)
    arch_mod.volatility = GARCH(1,0,1)
    arch_mod.distribution = StudentsT()
    res = arch_mod.fit(update_freq=5,disp='off')
    mu = model.params[0]
    theta = model.params[1]

    omega = res.params[1]
    alpha = res.params[2]
    beta = res.params[3]
    sigma_t = res.conditional_volatility.ix[-1]
    #print(res.conditional_volatility)
    sigma_predict = np.sqrt(omega + alpha * res.resid.ix[-1] ** 2 + beta * sigma_t ** 2)
    epsilon_t = sigma_t * np.random.standard_normal()
    epsilon_predict = sigma_predict * np.random.standard_normal()
    return_predict = mu + epsilon_predict + theta * epsilon_t
    print(return_predict)
    #测试2018年数据

    #按均值方程、波动率方程计算收益率
    t = 300
    price_pool1 = []
    price_pool2 = []
    for date in pd.date_range("2018-01-01","2018-03-31"):
        df2 = df1[date - dt.timedelta(t):date]
        model = sm.tsa.ARMA(df2, (0, 1)).fit(disp=0)
        arch_mod = ConstantMean(df2)
        arch_mod.volatility = GARCH(1, 0, 1)
        arch_mod.distribution = StudentsT()
        res = arch_mod.fit(update_freq=10, disp='off')
        #print('验证',res.conditional_volatility.ix[0])
        mu = model.params[0]
        theta = model.params[1]

        omega = res.params[1]
        alpha = res.params[2]
        beta = res.params[3]
        sigma_t = res.conditional_volatility.ix[-1]
        #epsilon_t = sigma_t * np.random.standard_normal()
        epsilon_t = res.resid.ix[-1]
        sigma_predict = np.sqrt(omega + alpha * epsilon_t ** 2 + beta * sigma_t ** 2)
        epsilon_predict1 = np.abs(sigma_predict * np.random.standard_normal())
        epsilon_predict2 = - np.abs(sigma_predict * np.random.standard_normal())
        return_predict1 = mu + epsilon_predict1 + theta * epsilon_t
        return_predict2 = mu + epsilon_predict2 + theta * epsilon_t

        price_pool1.append(return_predict1)
        price_pool2.append(return_predict2)



    _return1 = pd.DataFrame(price_pool1, index=df['2018-01-01':'2018-03-31'].index, columns=['up'])
    _return2 = pd.DataFrame(price_pool2, index=df['2018-01-01':'2018-03-31'].index, columns=['down'])
    #_price = np.exp(_return)
    origin = pd.DataFrame(df1["2018-01-01":"2018-03-31"], columns=['close_p'])
    origin['return1'] = _return1['up']
    origin['return2'] = _return2['down']
    origin['tag'] = (origin['close_p'] >= origin['return2']) & (origin['close_p'] <= origin['return1'])

    print(len(origin[origin['tag']==True]), len(origin))

    #print(flag,len(origin))

    #print(max(origin))
    #print(max(_price))
    #MSE = ((_return - origin) ** 2).mean()
    #print('MSE:',t,MSE)
    plt.plot(_return1, color='red')
    plt.plot(_return2, color='yellow')
    plt.plot(origin['close_p'])
    plt.legend()
    plt.show()