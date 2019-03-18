#-*- coding:utf-8 -*-
import pandas as pd
from database import engine_test, engine_exchange, engine_local,engine_quant
import datetime as dt
from matplotlib import pyplot as plt
import math
from sklearn import linear_model
import numpy as np
import time
import datetime as dt
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import r2_score, roc_auc_score, confusion_matrix
import itertools
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import svm
from sklearn import naive_bayes
engine = engine_quant()

def data():
    sql = '''select date,pair,low,high,close,volume from kline_1d where from_unixtime(date) >= "2017-11-01" and pair = 'BTC-USDT' order by date'''
    df_coin = pd.read_sql(sql, engine)
    df_coin['date'] = df_coin['date'].apply(
        lambda x: dt.datetime(time.localtime(int(x)).tm_year, time.localtime(int(x)).tm_mon,
                              time.localtime(int(x)).tm_mday))
    df_coin.index = df_coin['date']
    #进行时间区间的调整
    # low = df_coin['low'].resample('W').min()
    # high = df_coin['high'].resample('W').max()
    # close = df_coin['close'].resample('W').last()
    # volume = df_coin['volume'].resample('W').sum()
    # df_coin = pd.concat([low, high, close, volume], axis=1)
    # df_coin['pair'] = 'BTC-USDT'
    return df_coin

def RSI(t,df):
    df = df.copy()
    df['increase'] = df['close'] / df['close'].shift(1) - 1
    df['rsi'] = np.nan
    columns = list(df.columns)
    index = columns.index('rsi')

    for i in range(t-1,len(df)):
        df_window = df.iloc[i-t+1:i+1]
        df_up = df_window[df_window['increase'] > 0]
        #df_tie = df_window[df_window['increase'] == 0]
        df_down = df_window[df_window['increase'] < 0]

        up = df_up['increase'].mean()
        down = df_down['increase'].mean()
        rsi = round(up / (up - down) * 100, 2)
        df.iloc[i, index] = rsi

    df = df[['rsi']]
    return df

def PSY(t, m, df):
    df = df.copy()
    df['increase'] = df['close'] / df['close'].shift(1) - 1
    df['psy'] = np.nan
    columns = list(df.columns)
    index = columns.index('psy')

    for i in range(t-1,len(df)):
        df_window = df.iloc[i-t+1:i+1]
        df_up = df_window[df_window['increase'] > 0]
        #df_tie = df_window[df_window['increase'] == 0]
        psy = round(len(df_up) / t * 100, 2)
        df.iloc[i, index] = psy

    df['psyma'] = df['psy'].rolling(m).mean()
    df['gold'] = (df['psy'].shift(1) < df['psyma'].shift(1)) & (df['psy'] > df['psyma'])
    df['dead'] = (df['psy'].shift(1) > df['psyma'].shift(1)) & (df['psy'] < df['psyma'])
    df = df[['psy']]

    return df

def VR(t, df):
    df = df.copy()
    df['increase'] = df['close'] / df['close'].shift(1) - 1
    df['vr'] = np.nan
    columns = list(df.columns)
    index = columns.index('vr')

    for i in range(t-1,len(df)):
        df_window = df.iloc[i-t+1:i+1]
        df_up = df_window[df_window['increase'] > 0]
        df_tie = df_window[df_window['increase'] == 0]
        df_down = df_window[df_window['increase'] < 0]
        vr = round((df_up['volume'].sum() + 0.5 * df_tie['volume'].sum()) / (
                    df_down['volume'].sum() + 0.5 * df_tie['volume'].sum()) * 100, 2)
        df.iloc[i, index] = vr

    df = df[['vr']]
    return df

def volatility(t, df):
    df = df.copy()
    df['increase'] = df['close'] / df['close'].shift(1) - 1
    df['volatility'] = df['increase'].rolling(t).std()

    return df

def momentum(t, df):
    df = df.copy()
    df['momentum'] = df.close.rolling(t).apply(lambda x: round((x[-1] / x[0] - 1)*100,2), raw=False)

    return df

def ma(t, df):
    df = df.copy()
    df['ma'] = df.close.rolling(t).mean()
    return df

def MACD(df, t1=12, t2=26, t3=9):
    df = df.copy()
    ema1 = df.close.ewm(span=t1, adjust=True).mean()
    ema2 = df.close.ewm(span=t2, adjust=True).mean()
    dif = ema1 - ema2
    df['dif'] = dif
    dea = df.dif.ewm(span=t3, adjust=True).mean()
    df['macd'] = df.dif - dea

    return df

def bolling_band(df, t=20):
    df = df.copy()
    df['ma'] = df.close.rolling(t).mean()
    df['std'] = df.close.rolling(t).std()
    df['upper'] = df['ma'] + 2 * df['std']
    df['lower'] = df['ma'] - 2 * df['std']
    return df

def nasdq(df):
    data = pd.read_csv('d:/data_science/nadq.csv')
    data['Date'] = pd.to_datetime(data['Date'])
    data.index = data.Date
    del data['Date']
    date_list = pd.date_range(data.index[0], data.index[-1], freq='D')
    data = data.reindex(index=date_list, method='ffill')
    data = data.copy()
    data['profit'] = data['Adj Close'] / data['Adj Close'].shift(1) - 1
    data = data[data.index >= df.index[0]]
    return data

def usdt(df):
    data = pd.read_sql('select day, close_p, volume from qy_coin_data where coin_id = 1343', engine_test())
    data.index = pd.to_datetime(data.day)
    del data['day']
    date_list = pd.date_range(data.index[0], data.index[-1], freq='D')
    data = data.reindex(index=date_list, method='ffill')
    data = data.copy()
    data['profit'] = data.close_p / data.close_p.shift(1) - 1
    data = data[data.index >= df.index[0]]
    return data

def model_train():
    df = data()
    pairs = df['pair'].unique()
    df_new = pd.DataFrame()

    for pair in pairs:
        df_total = pd.DataFrame(columns=['pair', 'RSI', 'PSY',
                                         'VR', 'volatility', 'profit30',
                                         'ma7', 'ma28', 'macd', 'upper',
                                         'lower', 'nasdq', 'usdt_p', 'usdt_v'])
        df_coin = df[df['pair'] == pair]
        df_rsi = RSI(12, df_coin)
        df_psy = PSY(12,6, df_coin)
        df_vr = VR(12, df_coin)
        df_volatility = volatility(12, df_coin)
        df_momentum = momentum(12, df_coin)
        df_ma7 = ma(7, df_coin)
        df_ma28 = ma(28, df_coin)
        df_macd = MACD(df_coin)
        df_upper_lower = bolling_band(df_coin)
        df_nasdq = nasdq(df_coin)
        df_usdt = usdt(df_coin)
        df_total['momentum'] = df_momentum['momentum']
        df_total['volatility'] = df_volatility['volatility']
        df_total['RSI'] = df_rsi['rsi']
        df_total['PSY'] = df_psy['psy']
        df_total['VR'] = df_vr['vr']
        df_total['profit30'] = df_coin['close'].shift(-1) / df_coin['close'] > 1
        df_total['ma7'] = df_ma7['ma']
        df_total['ma28'] = df_ma28['ma']
        df_total['macd'] = df_macd['macd']
        df_total['upper'] = df_upper_lower['upper']
        df_total['lower'] = df_upper_lower['lower']
        df_total['nasdq'] = df_nasdq['profit']
        df_total['usdt_p'] = df_usdt['profit']
        df_total['usdt_v'] = df_usdt['volume']
        # avg = df_total['profit30'].mean()
        # std = df_total['profit30'].std()
        # df_total['profit30'] = (df_total['profit30'] - avg) / std
        df_total['pair'] = pair
        df_new = pd.concat([df_new,df_total], axis=0, join='outer')

    df_new.dropna(inplace=True)
    #print(df_new[['RSI', 'VR', 'volatility', 'momentum', 'ma7', 'ma28', 'macd', 'upper', 'lower', 'nasdq', 'profit30']].corr('pearson'))
    #标准化、降维
    # df_new['momentum'].hist()
    # plt.show()
    Scaler = StandardScaler()
    # train = Scaler.fit_transform(df_new[['RSI', 'VR', 'volatility', 'momentum', 'ma7', 'ma28', 'macd', 'upper', 'lower', 'nasdq']])
    # pca = PCA(n_components=10)
    # train_pca = pca.fit_transform(train)
    # # df_new['pca1'] = train_pca[:, 0]
    # # df_new['pca2'] = train_pca[:, 1]
    # # df_new['pca3'] = train_pca[:, 2]
    # # print(df_new)
    # X = np.matrix(train_pca)
    # VIF_list = [variance_inflation_factor(X, i) for i in range(X.shape[1])]
    # print(VIF_list)
    for param in range(100,1100, 100):
        RF = RandomForestClassifier(n_estimators=param, n_jobs=-1, oob_score=True)
        # RF = svm.SVC(kernel='poly')
        # RF = naive_bayes.GaussianNB()
        #RF = RandomForestRegressor(n_estimators=100, n_jobs=-1, oob_score=True)

        df_test = df_new.iloc[300:]
        # df_train = df_new.iloc[0:300]
        # train = Scaler.fit_transform(df_train[['RSI', 'VR', 'momentum', 'ma7', 'ma28', 'macd', 'upper', 'lower', 'usdt_p', 'usdt_v']])
        # # pca = PCA(n_components=10)
        # # train_pca = pca.fit_transform(train)
        # test = Scaler.fit_transform(df_test[['RSI', 'VR', 'momentum', 'ma7', 'ma28', 'macd', 'upper', 'lower', 'usdt_p', 'usdt_v']])
        # # test_pca = pca.fit_transform(test)
        # # RF.fit(train_pca, df_train['profit30'])
        # RF.fit(train, df_train['profit30'])
        # # predict = RF.predict(test_pca)
        # # predict = RF.predict(test)
        # print(RF.feature_importances_, RF.oob_score_)

        for k in range(1):
            predict = []
            for i in range(300, len(df_new)):
                #新增归一化及降维
                df_final = df_new.iloc[i-300:i+1]
                train = Scaler.fit_transform(df_final[['RSI', 'VR', 'volatility', 'momentum', 'ma7', 'ma28', 'macd', 'upper', 'lower', 'nasdq', 'usdt_p', 'usdt_v']])
                pca = PCA(n_components=6)
                train_pca = pca.fit_transform(train)
                # #RF.fit(df_final[0:-14][['RSI', 'VR', 'volatility', 'momentum', 'ma7', 'ma28', 'macd', 'upper', 'lower']], df_final.iloc[0:-14]['profit30'])
                # RF.fit(train[0:-7,:], df_final[0:-7]['profit30'])
                RF.fit(train_pca[0:-1,:], df_final.iloc[0:-1]['profit30'])
                #temp = RF.predict(np.array(df_new.iloc[i][['RSI', 'VR', 'volatility', 'momentum', 'ma7', 'ma28', 'macd', 'upper', 'lower']]).reshape(1, -1))
                temp = RF.predict(train_pca[-1].reshape(1,-1))
                predict.append(temp[0])

            df_test = df_test.copy()
            df_test['profit_predict'] = predict
            df_test['correct'] = (df_test['profit_predict'] == df_test['profit30'])
            print('correctivity: %f' % (df_test['correct'].sum() / len(df_test)))
            cm = confusion_matrix(df_test['profit30'], df_test['profit_predict'])
            print(cm)
        # plot_confusion_matrix(cm, classes=['down','up'])
    # print(df_test['profit30'].sum(), len(df_test))

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    model_train()
