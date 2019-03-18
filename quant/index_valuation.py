from database import engine_test
import pandas as pd


def twitter_index_value():
    df = pd.read_sql('select a.*,b.name,b.day7 from qy_twitter a left join qy_coin_info b on a.coin_id = b.id order by b.marketcap limit 50',engine_test())
    followers = df['followers']
    tweets = df['tweets']
    reply = df['reply']
    forward = df['forward']
    likes = df['likes']
    profit = df['day7']
    df['r_f'] = df['tweets'] / df['likes']
    print(df.corr()['day7'])

if __name__ == '__main__':
    twitter_index_value()