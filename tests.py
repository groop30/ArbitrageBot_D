# import time
import pandas as pd
import key
# from decimal import Decimal
import bin_utils as modul
import byb_utils as modul_byb
# from os import path
import talib
import ccxt
import statsmodels.api as stat
import statsmodels.tsa.stattools as ts
import sqlalchemy as sql
from sqlalchemy import Column
from mysql.connector import connect, Error
import numpy as np
from binance.client import Client
import datetime


binance = ccxt.binanceusdm({
    'enableRateLimit': True,
    'apiKey': key.binanceAPI,
    'secret': key.binanceSecretAPI,
})

binance_api = Client(
    api_key=key.binanceAPI,
    api_secret=key.binanceSecretAPI
)

pd.options.mode.chained_assignment = None
tf_5m = 5 * 60
tf_1h = 60 * 60
tf_5m_str = '5m'


def connect_to_mysql():
    try:
        connection = connect(
            host="127.0.0.1",
            user='root',
            password=key.mysqlroot,
        )

        return connection
    except Error as e:
        print(e)
        return None


def connect_to_mysql_database(base_name):
    try:
        connection = connect(
            host="127.0.0.1",
            user='root',
            password=key.mysqlroot,
            database = base_name
        )

        return connection
    except Error as e:
        print(e)
        return None


def connect_to_sqlalchemy():
    engine = sql.create_engine(f'mysql+mysqlconnector://root:{key.mysqlroot}@127.0.0.1:3306/binance', echo = True)
    return engine


def create_database(name, connect):
    create_db_query = f'CREATE DATABASE IF NOT EXISTS {name}'
    if connect.is_connected():
        print('Connected to MySQL database')

    with connect.cursor() as cursor:
        cursor.execute(create_db_query)

    show_db_query = "SHOW DATABASES"
    with connect.cursor() as cursor:
        cursor.execute(show_db_query)
    for db in cursor:
        print(db)


def create_olhc_mysql_table(name, connect):
    query = f"""
                    CREATE TABLE IF NOT EXISTS {name}
                    (
                        startTime DateTime,
                        time timestamp,
                        open Float32,
                        high Float32,
                        low Float32,
                        close Float32
                    )
                    ENGINE = ReplacingMergeTree(datetime)
                    PARTITION BY toYear(datetime)
                    ORDER BY (symbol, datetime)
                    """
    with connect.cursor() as cursor:
        cursor.execute(query)
        connect.commit()


def get_statistic(data1, data2):
    # Perform ADF test on the closing prices of fetched data
    result = stat.OLS(data1, data2).fit()
    c_t = ts.adfuller(result.resid, maxlag=20)
    return c_t[1]


def get_open_orders(asset):
    # res = binance.fetch_positions([asset])
    # res = binance.fetch_my_trades(asset)
    # res = binance.fetch_closed_orders(asset)
    res = binance.fetch_orders(asset)
    df = pd.DataFrame([res])
    return df


def first_fullfil_checkdf():
    connection = modul.connect_to_sqlalchemy_binance()
    check_table = modul.create_check_table(connection)
    # query = check_table.select()
    filepath_close = r'.\reports\bin_to_check.csv'
    to_check_df = pd.read_csv(filepath_close, sep="\t")
    with connection.connect() as conn:
        to_check_df.to_sql(name='bin_to_check', con=conn, if_exists='append', index=False)
    print('первичная загрузка закончена!')


def first_fullfil_closedf():
    connection = modul.connect_to_sqlalchemy_binance()
    close_table = modul.create_close_table(connection)
    # query = close_table.select()
    filepath_close = r'.\reports\bin_to_close.csv'
    to_close_df = pd.read_csv(filepath_close, sep="\t")
    with connection.connect() as conn:
        to_close_df.to_sql(name='bin_to_close', con=conn, if_exists='append', index=False)
    print('первичная загрузка закончена!')


def first_fullfil_orders():
    connection = modul.connect_to_sqlalchemy_binance()
    orders_table = modul.create_orders_table(connection)
    # query = orders_table.select()
    filepath_close = r'.\reports\orders_log.csv'
    orders_df = pd.read_csv(filepath_close, sep="\t")
    with connection.connect() as conn:
        orders_df.to_sql(name='orders_log', con=conn, if_exists='append', index=False)
    print('первичная загрузка закончена!')


def all_futures_to_db():
    # all_futures = modul.get_all_futures()
    # all_futures.to_csv(r'.\screening\all_futures_binance.csv', index=False, sep="\t")
    # all_spot_binance= modul.get_all_spot_df('binance')
    # all_spot_binance.to_csv(r'.\screening\all_spot_binance.csv', index=False, sep="\t")
    all_futures_byb = modul_byb.get_all_futures_bybit()
    all_futures_byb.to_csv(r'.\screening\all_futures_byb.csv', index=False, sep="\t")
    all_spot_byb = modul_byb.get_all_spot_bybit()
    all_spot_byb.to_csv(r'.\screening\all_spot_byb.csv', index=False, sep="\t")


def add_result(df, result, position):
    # создаем строку с данными
    new_row = pd.DataFrame({
        'position': [position],
        'result': [round(result, 2)]
    },
        index=None)
    df = pd.concat([df, new_row], ignore_index=True)
    return df


def test_oc_strategy(coin1, coin2, startdate):
    # получение исходных данных
    connection = modul.connect_to_sqlalchemy_binance()
    end = datetime.datetime.now().timestamp()
    df_coin1 = modul.get_sql_history_price(coin1, connection, startdate,end)
    df_coin2 = modul.get_sql_history_price(coin2, connection, startdate, end)
    spread_df = modul.make_spread_df(df_coin1, df_coin2, last_to_end=False, tf=tf_5m)
    spread_df['bb1_up'], spread_df['sma'],spread_df['bb1_down'], = talib.BBANDS(spread_df.close, 240, 1, 1, 0)
    spread_df['bb3_up'], aaa, spread_df['bb3_down'] = talib.BBANDS(spread_df.close, 240, 3, 3, 0)
    spread_df['bb4_up'], bbb, spread_df['bb4_down'] = talib.BBANDS(spread_df.close, 240, 4.2, 4.2, 0)

    # определелим переменные для расчетов
    result_df = pd.DataFrame(columns=['position', 'result'])
    inPosition = False
    first_take_close = False
    second_take_close = False
    enterShort = 0.0
    enterLong = 0.0
    stopShort = 0.0
    stopLong = 0.0
    # result = 0.0
    for index in range(len(spread_df)):
        # вынем из дф нужные данные в переменные
        close = spread_df.iloc[index]['close']
        if index > 240:
            close_before = spread_df.iloc[index-1]['close']
        else:
            close_before = 0.0
        bb3_up = spread_df.iloc[index]['bb3_up']
        bb3_down = spread_df.iloc[index]['bb3_down']

        # сначала смотрим условия для открытия позиции
        if not inPosition:
            if close_before > bb3_up > close:
                # открываем позицию в шорт
                enterShort = close
                stopShort = spread_df.iloc[index]['bb4_up']
                inPosition = True
            elif close > bb3_down > close_before and close_before != 0:
                enterLong = close
                stopLong = spread_df.iloc[index]['bb4_down']
                inPosition = True
        else:
            # сначала проверяем на превышение риска
            if close > stopShort and stopShort != 0.0:
                # закрываемся
                inPosition = False
                result = (enterShort - close) / enterShort * 100
                result_df = add_result(result_df, result, 'stop loss')
                enterShort = stopShort = 0.0
                first_take_close = False
                second_take_close = False
            elif close < stopLong:
                inPosition = False
                result = (close - enterLong) / enterLong * 100
                result_df = add_result(result_df, result, 'stop loss')
                enterLong = stopLong = 0.0
                first_take_close = False
                second_take_close = False

            # если не отстопило, проверяем на условия закрытия по тейку
            else:
                bb1_up = spread_df.iloc[index]['bb1_up']
                bb1_down = spread_df.iloc[index]['bb1_down']
                sma = spread_df.iloc[index]['sma']
                if first_take_close:
                    if second_take_close:
                        # смотрим последний уровень закрытия
                        if close < bb1_down and enterShort > 0.0:
                            # полностью закрываем шорт
                            inPosition = False
                            result = ((enterShort - close) / enterShort * 100)/3
                            result_df = add_result(result_df, result, '3rd take')
                            enterShort = stopShort = 0.0
                            first_take_close = False
                            second_take_close = False
                        elif close > bb1_up and enterLong > 0.0:
                            inPosition = False
                            result = ((close - enterLong) / enterLong * 100)/3
                            result_df = add_result(result_df, result, '3rd take')
                            enterLong = stopLong = 0.0
                            first_take_close = False
                            second_take_close = False
                    else:
                        if close < sma and enterShort > 0.0:
                            # берем треть от профита, т.к. закрыли бы только треть объема
                            result = ((enterShort - close) / enterShort * 100)/3
                            # result = 0.0
                            result_df = add_result(result_df, result, '2nd take')
                            stopShort = bb1_up
                            second_take_close = True
                        elif close > sma and enterLong > 0.0:
                            result = ((close - enterLong) / enterLong * 100)/3
                            # result = 0.0
                            result_df = add_result(result_df, result, '2nd take')
                            stopLong = bb1_down
                            second_take_close = True
                else:
                    if close < bb1_up and enterShort > 0.0:
                        # берем треть от профита, т.к. закрыли бы только треть объема
                        result = ((enterShort - close) / enterShort * 100) / 3
                        # result = 0.0
                        result_df = add_result(result_df, result, '1st take')
                        first_take_close = True
                        stopShort = bb3_up
                    elif close > bb1_down and enterLong > 0.0:
                        result = ((close - enterLong) / enterLong * 100) / 3
                        # result = 0.0
                        result_df = add_result(result_df, result, '1st take')
                        first_take_close = True
                        stopLong = bb3_down

    max_loss = result_df['result'].min()
    total = result_df['result'].sum()
    print(result_df)
    print(max_loss)
    print(total)


def williams_fractals(df, n):
    high_prices = df['high']
    low_prices = df['low']

    up_fractals = []
    down_fractals = []

    for i in range(n, len(high_prices) - n):
        if all(high_prices[i - n:i] < high_prices[i]) and all(high_prices[i + 1:i + n + 1] < high_prices[i]):
            up_fractals.append(i)

        if all(low_prices[i - n:i] > low_prices[i]) and all(low_prices[i + 1:i + n + 1] > low_prices[i]):
            down_fractals.append(i)

    return up_fractals, down_fractals


def pivot_point_supertrend(coin1, start_date):
    import matplotlib.pyplot as plt

    connection = modul.connect_to_sqlalchemy_binance()
    end = datetime.datetime.now().timestamp()
    spread_df = modul.get_sql_history_price(coin1, connection, start_date, end)
    # df_coin2 = modul.get_sql_history_price(coin2, connection, start_date, end)
    # spread_df = modul.make_spread_df(df_coin1, df_coin2, last_to_end=False, tf=tf_5m)

    # Set up input variables
    prd = 2
    factor = 3
    pd_atr = 10

    # Get high and low prices
    high = spread_df['high']
    low = spread_df['low']
    close = spread_df['close']

    # Calculate pivots
    # pivot_high = talib.MAX(high.shift(1), timeperiod=prd)
    # pivot_low = talib.MIN(low.shift(1), timeperiod=prd)
    pivot_high, pivot_low = williams_fractals(spread_df, prd)

    # Calculate the center line using pivot points
    # center = (ph + pl) / 2
    pivot_high_series = spread_df.index.isin(pivot_high)
    pivot_low_series = spread_df.index.isin(pivot_low)

    pivot_high_values = np.where(pivot_high_series, spread_df['high'], False)
    pivot_low_values = np.where(pivot_low_series, spread_df['low'], False)

    all_pivots = np.where(pivot_high_values, pivot_high_values, pivot_low_values)
    pd.Series(all_pivots).replace(0, np.nan, inplace=True)
    # filled_center = pd.Series(center).fillna(method='ffill')
    # spread_df['center'] = pd.Series(filled_center).rolling(2).apply(lambda x: (x[-1] * 2 + x[0]) / 3, raw=True)
    spread_df['pivots'] = all_pivots
    spread_df['center'] = np.nan
    for i in range(1, len(all_pivots)):
        if not pd.isna(all_pivots[i]):
            if not pd.isna(spread_df.iloc[i-1]['center']):
                spread_df.at[i, 'center'] = (spread_df.iloc[i -1]['center'] * 2  + all_pivots[i]) / 3
            else:
                spread_df.at[i, 'center'] = all_pivots[i]
        else:
            if not pd.isna(spread_df.iloc[i - 1]['center']):
                spread_df.at[i, 'center'] = spread_df.iloc[i - 1]['center']
            else:
                spread_df.at[i, 'center'] = all_pivots[i]

    # Upper/lower bands calculation
    atr = talib.ATR(high, low, close, timeperiod=pd_atr)
    up = spread_df['center'] - (factor * atr)
    dn = spread_df['center'] + (factor * atr)

    # # Get the trend - подумать, как сделать без цикла...
    # trend = talib.MAX(close, timeperiod=prd)
    # trend = np.where(close < talib.MAX(close.shift(), timeperiod=prd), -1, trend)
    # trend = np.where(close < dn.shift(), -1, trend)
    # trend = np.where(close > up.shift(), 1, trend)
    spread_df['trend'] = np.nan
    spread_df['switch_to'] = False
    for i in range(pd_atr,len(spread_df)):
        if pd.isna(spread_df.iloc[i-1]['trend']):
            if spread_df.iloc[i]['close'] < up[i]:
                spread_df.at[i, 'trend'] = dn[i]
            else:
                spread_df.at[i, 'trend'] = up[i]
            spread_df.at[i, 'switch_to'] = True
        else:
            # смотрим предыдущие значения, что бы понять, какой был тренд
            if spread_df.iloc[i - 1]['trend'] > spread_df.iloc[i-1]['close']:
                # значит был тренд вниз
                if dn[i] >= spread_df.iloc[i]['close']:
                    # пробития тренда не было, значит тренд остается
                    spread_df.at[i, 'trend'] = dn[i]
                else:
                    # тренд пробит, меняем линию
                    spread_df.at[i, 'trend'] = up[i]
                    spread_df.at[i, 'switch_to'] = True
            elif spread_df.iloc[i - 1]['trend'] < spread_df.iloc[i-1]['close']:
                # значит был тренд вверх
                if up[i] <= spread_df.iloc[i]['close']:
                    # пробития тренда не было, значит тренд остается
                    spread_df.at[i, 'trend'] = up[i]
                else:
                    # тренд пробит, меняем линию
                    spread_df.at[i, 'trend'] = dn[i]
                    spread_df.at[i, 'switch_to'] = True
            else:
                # Нужно учесть ситуацию цены равной тренду, в этом случае смены тренда еще не происходит
                # поэтому нужно посмотреть на две свечи назад. Бывает ли такое? Пока просто отмечу, что бы не захламлять код
                print("произошла исключительная ситуация, добавь исключение в код")

    return spread_df


def calculate_central_line():
    prices = [10, 12, 15, 14, 13, 11, 9, 12, 14, 16]
    prd = 2
    pivot_high = pd.Series(prices).rolling(prd).max()
    pivot_low = pd.Series(prices).rolling(prd).min()
    center = np.where(pivot_high.notnull(), pivot_high, pivot_low)
    center = pd.Series(center).rolling(2).apply(lambda x: (x[-1] * 2 + x[0]) / 3, raw=True)
    return center.tolist()


# Example usage


# ##############################################################
# тестирую сверку позиций
# modul.find_lost_trades()
# modul.sql_table_to_csv('bin_to_check')
# modul.sql_table_to_csv('bin_to_close')
# modul.sql_table_to_csv('orders_log')

# update_closedf()
# first_fullfil_checkdf()
# first_fullfil_closedf()
# first_fullfil_orders()
# all_futures_to_db()

# ##############################################################
# тестирую подключение и работу с sql
# connection = connect_to_mysql()
# connection = connect_to_sqlalchemy()
# create_database('bybit', connection)
# result = modul.create_olhc_table('AAVEUSDT', connection)
#
# end_time = datetime.datetime.now().timestamp()
# start_time = datetime.datetime.now().timestamp() - 2000 * tf_5m
# df = modul.get_sql_history_price('AAVEUSDT', connection, start_time, end_time)
# print(df)
# ##############################################################
# тестирую коинтеграцию
# end_time = datetime.datetime.now().timestamp()
start_time = datetime.datetime.now().timestamp() - 2000 * tf_5m
# coin1 = modul.get_history_price('1000SHIBUSDT', start_time, end_time, tf_5m)
# coin2 = modul.get_history_price('RLCUSDT', start_time, end_time, tf_5m)
# spread_df = modul.make_spread_df(coin1, coin2, True, tf_5m)

# calculate_central_line()
# get_statistic(coin1['close'], coin2['close'])
# all_futures_to_db()
# test_oc_strategy('DGBUSDT', 'IOSTUSDT', start_time)
# calculate_central_line()
pivot_point_supertrend('DGBUSDT', start_time)