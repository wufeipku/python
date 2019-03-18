import pandas as pd
import requests
from database import engine_test
from matplotlib import pyplot as plt
import numpy as np

def data_clean():
    df = pd.read_sql('select coin_id,day,close_p,market_cap from qy_coin_data where day >= "2017-01-01" order by coin_id,day',engine_test())
    coin_ids = list(set(df.coin_id))
    coin_ids = coin_ids[0:100]
    df_new = pd.DataFrame()

    for id in coin_ids:
        df_coin = df[df.coin_id == id]
        if len(df_coin) < 180:
            continue
        df_coin_copy = df_coin.loc[:, :]
        df_coin_copy['return'] = df_coin_copy.close_p / df_coin_copy.close_p.shift(1)
        df_coin_copy = df_coin_copy.dropna()
        #df_coin_copy.fillna(0, inplace=True)
        #df_coin_copy['std_down'] = 0
        df_coin_copy['std'] = 0.0
        df_coin_copy['return_1mon'] = 0.0

        for i in range(1, len(df_coin_copy)):
            if i <= 90:
                df_coin_copy.iloc[i]['std'] = df_coin_copy.iloc[1:i+1]['return'].std()
            else:
                df_coin_copy.iloc[i]['std'] = df_coin_copy.iloc[i-89:i+1]['return'].std()

            if i >= len(df_coin_copy) - 30:
                df_coin_copy.iloc[i]['return_1mon'] = df_coin_copy.iloc[len(df_coin_copy)-1]['close_p'] / df_coin_copy.iloc[i]['close_p'] - 1
            else:
                df_coin_copy.iloc[i]['return_1mon'] = df_coin_copy.iloc[i+30]['close_p'] / df_coin_copy.iloc[i]['close_p'] - 1

        df_new = pd.concat([df_new,df_coin_copy], join='outer')
        #df_new = df_new.sort_values(by='day', ascending = True)
    df_final = df_new.loc[:, ['std', 'return_1mon']]
    plt.scatter(df_final['std'], df_final['return_1mon'])
    plt.show()

if __name__ == '__main__':
    data_clean()


