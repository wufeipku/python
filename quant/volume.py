from database import engine_test,engine_smyt
import pandas as pd
from matplotlib import pyplot as plt

if __name__ == '__main__':
    df = pd.read_sql('select * from qy_coin_data where day between date_sub(curdate(),interval 3 month) and curdate() and coin_id = 1 order by day',engine_test())
    day = df['day'].tolist()
    volume = df['volume'].tolist()
    activity = (df['volume'] / df['market_cap']).tolist()
    df['day'] = pd.to_datetime(df['day'])
    df.index = df['day']
    print(df.resample('Q').last())
    # df = pd.read_sql('select * from coin_trading_copy where statistic_date between date_sub(curdate(),interval 3 month) and curdate() and coin_id = "JRB100001" order by statistic_date',engine_smyt())
    # day = df['statistic_date'].tolist()
    # volume = df['turnover_usd'].tolist()
    # activity = (df['turnover_usd'] / df['market_value_usd']).tolist()


    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.bar(day,volume,color='blue',label='volume')
    ax2 = ax1.twinx()
    ax2.plot(day,activity,color='red',label='activity')
    plt.show()

    print(volume)
    print(activity)
    print(day)