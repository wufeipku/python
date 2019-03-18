from database import engine_test
import pandas as pd

def meanprofit():
    df = pd.read_sql('select * from qy_coin_data having(day > "2017-01-01") order by day ', engine_test())
    coin_ids = pd.read_sql('select id from qy_coin_info order by marketcap desc limit 500', engine_test())['id'].tolist()

    for coin_id in coin_ids:
        df_coin = df[df.coin_id == coin_id]
        if len(df_coin) < 100:
            continue
        df_coin = df_coin.copy()
        df_coin['profit'] = df_coin['close_p'] / df_coin['close_p'].shift(30) - 1
        df_coin['future'] = df_coin['close_p'].shift(-30) / df_coin['close_p'] - 1
        coef = df_coin[['profit','future']].corr().loc['profit','future']
        print(coin_id, coef)

if __name__ == '__main__':
    meanprofit()