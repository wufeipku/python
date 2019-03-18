import pandas as pd
from database import engine_test
import requests


def main():
    #engine = engine_test()
    #df = pd.read_sql('select id, name from qy_coin_info',engine)
    headers = {'X-CMC_PRO_API_KEY': '9060432f-9802-49f4-8f52-ee30fadcb2dd'}
    df_id = requests.get('https://pro-api.coinmarketcap.com/v1/cryptocurrency/map',headers=headers)
    df_id = df_id.json()['data'][5000:]
    print(df_id)
    id_list = []

    for id in df_id:
        id_list.append(','+str(id['id']))

    ids = ''.join(id_list)[1:]
    print(type(ids))

    df_info = requests.get('https://pro-api.coinmarketcap.com/v1/cryptocurrency/info',params={'id': ids},headers=headers)
    print(df_info)
    print(df_info.json())

if __name__ == '__main__':
    main()