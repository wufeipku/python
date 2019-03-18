from sqlalchemy import create_engine


db = {

    'test':{
        'NAME': 'digdig_io',
        'USER': 'root',
        'PASSWORD': 'Digdig@I0',
        'HOST': '106.75.115.242',
        'PORT': 3306,
    },
    'online': {
        'NAME': 'digdig_io',
        'USER': 'op',
        'PASSWORD': 'op@123.',
        'HOST': '60.205.223.152',
        'PORT': 3306,
    },
    'smyt':{
        'NAME': 'currencies',
        'USER': 'cur_read2',
        'PASSWORD': '2tF6YSq45C43',
        'HOST': 'sh-cdb-s089fj1s.sql.tencentcdb.com',
        'PORT': 63405,
    },
    'foreign':{
        'NAME': 'digdig_exchange',
        'USER': 'root',
        'PASSWORD': 'Hub@x',
        'HOST': '45.77.157.110',
        'PORT': 3309,
    },
    'exchange':{
        'NAME': 'digdig_exchange',
        'USER': 'root',
        'PASSWORD': 'Hub@x',
        'HOST': '106.75.115.242',
        'PORT': 3308,
    },
    'local':{
        'NAME': 'digdig',
        'USER': 'root',
        'PASSWORD': 'wf@hbcj123',
        'HOST': 'localhost',
        'PORT': 3306,
    },
    'quant': {
        'NAME': 'quant',
        'USER': 'quant',
        'PASSWORD': 'quant',
        'HOST': '149.28.239.9',
        'PORT': 4406,
    },
}

def engine_test():
    _engine = create_engine("mysql+pymysql://{}:{}@{}:{}/{}".format(
        db['test']['USER'],
        db['test']['PASSWORD'],
        db['test']['HOST'],
        db['test']['PORT'],
        db['test']['NAME'],
    ), connect_args={"charset": "utf8"})
    return _engine

def engine_foreign():
    _engine = create_engine("mysql+pymysql://{}:{}@{}:{}/{}".format(
        db['foreign']['USER'],
        db['foreign']['PASSWORD'],
        db['foreign']['HOST'],
        db['foreign']['PORT'],
        db['foreign']['NAME'],
    ), connect_args={"charset": "utf8"})
    return _engine

def engine_exchange():
    _engine = create_engine("mysql+pymysql://{}:{}@{}:{}/{}".format(
        db['exchange']['USER'],
        db['exchange']['PASSWORD'],
        db['exchange']['HOST'],
        db['exchange']['PORT'],
        db['exchange']['NAME'],
    ), connect_args={"charset": "utf8"})
    return _engine

def engine_online():
    _engine = create_engine("mysql+pymysql://{}:{}@{}:{}/{}".format(
        db['online']['USER'],
        db['online']['PASSWORD'],
        db['online']['HOST'],
        db['online']['PORT'],
        db['online']['NAME'],
    ), connect_args={"charset": "utf8"})
    return _engine

def engine_smyt():
    _engine = create_engine("mysql+pymysql://{}:{}@{}:{}/{}".format(
        db['smyt']['USER'],
        db['smyt']['PASSWORD'],
        db['smyt']['HOST'],
        db['smyt']['PORT'],
        db['smyt']['NAME'],
    ), connect_args={"charset": "utf8"})
    return _engine

def engine_local():
    _engine = create_engine("mysql+pymysql://{}:{}@{}:{}/{}".format(
        db['local']['USER'],
        db['local']['PASSWORD'],
        db['local']['HOST'],
        db['local']['PORT'],
        db['local']['NAME'],
    ), connect_args={"charset": "utf8"})
    return _engine

def engine_quant():
    _engine = create_engine("mysql+pymysql://{}:{}@{}:{}/{}".format(
        db['quant']['USER'],
        db['quant']['PASSWORD'],
        db['quant']['HOST'],
        db['quant']['PORT'],
        db['quant']['NAME'],
    ), connect_args={"charset": "utf8"})
    return _engine