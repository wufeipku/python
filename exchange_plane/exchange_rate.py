from database import engine_local,engine_test
import pandas as pd
import requests
import datetime as dt

engine = engine_test()

def exchange_rate():

    api_key = 'e8a15a9c63d4f47e3d019d1a45c9be78'
    url = 'http://data.fixer.io/api/'
    date_list = list(pd.date_range('2014-01-01','2014-12-31'))
    for date in date_list:
        time = date.strftime('%Y-%m-%d')
        re = requests.get(url+time+"?access_key={}".format(api_key)+'&symbols=CNY,USD,FRF,HKD,CHF,CAD,GBP,NLG,DEM,BEF,JPY,AUD')
        re = re.json()
        symbols = re['rates']
        print(symbols)
        eur = round(1 / symbols['USD'],6)
        base = symbols['USD']

        for key in symbols.keys():
            if key != 'USD':
                symbols[key] = symbols[key] / base
                try:
                    engine.execute("replace into qy_exchange_rate(date,symbol,rate) values('{}','{}',{})".format(re['date'],key,symbols[key]))

                except Exception as e:
                    print(e)

        try:
            engine.execute("replace into qy_exchange_rate(date,symbol,rate) values('{}','{}',{})".format(re['date'], 'EUR', eur))
        except Exception as e:
            print(e)

        print(re['date'])

    return

def exchange_rate_daily():

    api_key = 'e8a15a9c63d4f47e3d019d1a45c9be78'
    url = 'http://data.fixer.io/api/'
    date = dt.date.today()
    re = requests.get(
        url + date.strftime('%Y-%m-%d') + "?access_key={}".format(api_key) + '&symbols=CNY,USD,FRF,HKD,CHF,CAD,GBP,NLG,DEM,BEF,JPY,AUD')
    re = re.json()
    symbols = re['rates']
    print(symbols)
    eur = round(1 / symbols['USD'], 6)
    base = symbols['USD']

    for key in symbols.keys():
        if key != 'USD':
            symbols[key] = symbols[key] / base
            try:
                engine.execute(
                    "replace into qy_exchange_rate(date,symbol,rate) values('{}','{}',{})".format(re['date'], key,
                                                                                              symbols[key]))

            except Exception as e:
                print(e)

    try:
        engine.execute(
            "replace into qy_exchange_rate(date,symbol,rate) values('{}','{}',{})".format(re['date'], 'EUR', eur))
    except Exception as e:
        print(e)

if __name__ == '__main__':
    #bitfinex_kline_table()
    #exchange_rate_table()
    exchange_rate()
    #exchange_currency_table()
    #exchange_rate_daily()