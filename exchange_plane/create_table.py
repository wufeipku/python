#-*- coding:utf-8 -*-
from database import engine_local,engine_test
import pandas as pd
import pymysql

def coin_index_table():
    engine = engine_local()
    sql = '''create table if not exists coin_index(
    id int not null primary key auto_increment,
    `date` date not null unique ,
    coin_index decimal(10,4) null,
    coin_index_return decimal(10,4) null
    )
    '''
    engine.execute(sql)

    return

def bitfinex_kline_table():
    engine = engine_local()
    sql = '''create table if not exists kline_1m(
        coin_id varchar(20) not null,
        date datetime not null,
        exchange varchar(20) not null,
        open decimal(20,9) null,
        close decimal(20,9) null,
        high decimal(20,9) null,
        low decimal(20,9) null,
        vol decimal(20,4) null,
        primary key(coin_id,date,exchange
        )
              '''
    cur = engine.execute(sql)

    return

def exchange_currency_table():
    engine = engine_local()
    sql = '''create table if not exists exchange_currency(
      id int primary key auto_increment not null,
      exchange varchar(20) not null,
      date date not null,
      name varchar(50) not null,
      symbol varchar(20) not null )
    '''
    engine.execute(sql)
    return


def exchange_rate_table():
    engine = engine_test()
    sql = '''create table if not exists qy_exchange_rate(
      id int primary key auto_increment not null,
      symbol varchar(6) not null,
      date date not null,
      rate decimal(10,6) null,
    )
    '''
    cur = engine.execute(sql)
    return

def depth_table():
    engine = engine_local()
    sql = ''' create table if not exists depth(
    coin_id varchar(20) not null,
    pair varchar(20) not null,
    exchange varchar(20) not null,
    date bigint not null,
    ask1_p decimal(20,10) null,
    ask2_p decimal(20,10) null,
    ask3_p decimal(20,10) null,
    ask4_p decimal(20,10) null,
    ask5_p decimal(20,10) null,
    ask6_p decimal(20,10) null,
    ask7_p decimal(20,10) null,
    ask8_p decimal(20,10) null,
    ask9_p decimal(20,10) null,
    ask10_p decimal(20,10) null,
    ask1_v decimal(20,10) null,
    ask2_v decimal(20,10) null,
    ask3_v decimal(20,10) null,
    ask4_v decimal(20,10) null,
    ask5_v decimal(20,10) null,
    ask6_v decimal(20,10) null,
    ask7_v decimal(20,10) null,
    ask8_v decimal(20,10) null,
    ask9_v decimal(20,10) null,
    ask10_v decimal(20,10) null,
    bid1_p decimal(20,10) null,
    bid2_p decimal(20,10) null,
    bid3_p decimal(20,10) null,
    bid4_p decimal(20,10) null,
    bid5_p decimal(20,10) null,
    bid6_p decimal(20,10) null,
    bid7_p decimal(20,10) null,
    bid8_p decimal(20,10) null,
    bid9_p decimal(20,10) null,
    bid10_p decimal(20,10) null,
    bid1_v decimal(20,10) null,
    bid2_v decimal(20,10) null,
    bid3_v decimal(20,10) null,
    bid4_v decimal(20,10) null,
    bid5_v decimal(20,10) null,
    bid6_v decimal(20,10) null,
    bid7_v decimal(20,10) null,
    bid8_v decimal(20,10) null,
    bid9_v decimal(20,10) null,
    bid10_v decimal(20,10) null,
    primary key(pair,exchange,date));
    '''
    engine.execute(sql)

    return

if __name__ == '__main__':
    #bitfinex_kline_table()
    #exchange_rate_table()
    #exchange_rate()
    #exchange_currency_table()
    #exchange_rate_daily()

    coin_index_table()

    # engine = engine_local()
    # data = pd.read_sql('select * from qy_exchange_rate',engine)
    # inst_data = []
    # engine = engine_test()
    # for i in range(len(data)):
    #     sql = '''insert into qy_exchange_rate(symbol,date,rate) values ('{}','{}',{})
    #         '''.format(data.iloc[i][1],data.iloc[i][2],data.iloc[i][3])
    #     engine.execute(sql)


    # con = pymysql.connect(host='39.107.248.189',user='root',passwd='Digdig@I0',port=6666,db='digdig_io')
    # cur = con.cursor()
    # cur.executemany(sql,inst_data)
    # con.commit()
    # cur.close()
