from sqlalchemy import create_engine

def engine_local():
    _engine = create_engine("mysql+pymysql://{}:{}@{}:{}/{}".format(
        'root',
        'wf@hbcj123',
        'localhost',
        '3306',
        'digdig',
    ), connect_args={"charset": "utf8"})

    return _engine

def engine_test():
    _engine = create_engine("mysql+pymysql://{}:{}@{}:{}/{}".format(
        'root',
        'Digdig@I0',
        '106.75.115.242',
        '3306',
        'digdig_io',
    ), connect_args={"charset": "utf8"})

    return _engine

def engine_online():
    _engine = create_engine("mysql+pymysql://{}:{}@{}:{}/{}".format(
        'op',
        'op@123.',
        '60.205.223.152',
        '3306',
        'digdig_io',
    ), connect_args={"charset": "utf8"})

    return _engine