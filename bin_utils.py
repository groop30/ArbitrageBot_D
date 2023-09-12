import key
import requests
import ccxt
import time
import pandas as pd
import datetime
from pathlib import Path
from os import path
from typing import Optional
from binance.client import Client
from binance.enums import HistoricalKlinesType
import talib
import statsmodels.api as stat
import statsmodels.tsa.stattools as ts
import numpy as np
from statsmodels.tsa.vector_ar.vecm import coint_johansen
# import plotly.graph_objects as go
import sqlalchemy as sql
from sqlalchemy import bindparam
from decimal import Decimal
import indicators as ind

binance = ccxt.binanceusdm({
    'enableRateLimit': True,
    'apiKey': key.binanceAPI,
    'secret': key.binanceSecretAPI,
})
binance2 = ccxt.binanceusdm({
    'enableRateLimit': True,
    'apiKey': key.binanceAPI2,
    'secret': key.binanceSecretAPI2,
})
binance_api = Client(
    api_key=key.binanceAPI,
    api_secret=key.binanceSecretAPI
)
bybit = ccxt.bybit({
    'apiKey': key.bybitAPI,
    'secret': key.bybit_secretAPI,
})

pd.options.mode.chained_assignment = None

tf_5m = 5 * 60
tf_1m = 60
tf_1h = 60 * 60
tf_5m_str = '5m'
tf_1m_str = '1m'

exception_list = ['BTSUSDT', 'SCUSDT', 'TLMUSDT', 'BTCSTUSDT', 'FTTUSDT',
                  'XEMUSDT', 'XLMUSDT', 'SRMUSDT', 'RAYUSDT', 'CTSIUSDT',
                  'CHRUSDT', 'FLOWUSDT', 'SUSHIUSDT', 'CVCUSDT', 'BNXUSDT']

TELEGRAM_TOKEN = key.Black_Bulls_bot_API
TELEGRAM_CHANNEL = key.TELEGRAM_CHANNEL


def send_message_to_telegram(text, channel):

    if channel == 1:
        try:
            requests.get('https://api.telegram.org/bot{}/sendMessage'.format(TELEGRAM_TOKEN), params=dict(
                         chat_id='@blackbullssignals', text=text))
        except ValueError:
            print(f'Не получилась отправка сообщения в Telegram - {text}')
    elif channel == 2:
        try:
            requests.get('https://api.telegram.org/bot{}/sendMessage'.format(key.team2_alerts_bot), params=dict(
                         chat_id='@team2alerts', text=text))
        except ValueError:
            print(f'Не получилась отправка сообщения в Telegram - {text}')


# ######################################################################
# Процедуры для работы с SQL
#
# ######################################################################
def connect_to_sqlalchemy_binance():
    engine = sql.create_engine(f'mysql+mysqlconnector://root:{key.mysqlroot}@127.0.0.1:3306/binance', echo=False)
    return engine


def connect_to_sqlalchemy_bybit():
    engine = sql.create_engine(f'mysql+mysqlconnector://root:{key.mysqlroot}@127.0.0.1:3306/bybit', echo=False)
    return engine


def create_olhc_table(coin, connect):
    meta = sql.MetaData()

    olhc_table = sql.Table(
        coin, meta,
        sql.Column('time', sql.BIGINT, primary_key=True),
        sql.Column('startTime', sql.DateTime),
        sql.Column('open', sql.Float),
        sql.Column('high', sql.Float),
        sql.Column('low', sql.Float),
        sql.Column('close', sql.Float),
    )
    meta.create_all(connect)
    return olhc_table


def create_close_table(connect):
    meta = sql.MetaData()

    close_table = sql.Table(
        'bin_to_close', meta,
        sql.Column('coin1_id', sql.BIGINT),
        sql.Column('coin2_id', sql.BIGINT),
        sql.Column('pair', sql.Text),
        sql.Column('coin1', sql.Text),
        sql.Column('coin2', sql.Text),
        sql.Column('going_to', sql.Text),
        sql.Column('price', sql.Float),
        sql.Column('stop', sql.Float),
        sql.Column('size1', sql.Float),
        sql.Column('size2', sql.Float),
        sql.Column('strategy', sql.Text),
        sql.Column('lookback', sql.BIGINT),
        sql.Column('up', sql.Float),
        sql.Column('down', sql.Float),
        sql.Column('pnl', sql.Float),
        sql.Column('exchange', sql.Text),
        sql.Column('c1_op_price', sql.Float),
        sql.Column('c2_op_price', sql.Float),
    )
    meta.create_all(connect)
    return close_table


def create_check_table(connect):
    meta = sql.MetaData()

    check_table = sql.Table(
        'bin_to_check', meta,
        sql.Column('pair', sql.Text),
        sql.Column('coin1', sql.Text),
        sql.Column('coin2', sql.Text),
        sql.Column('strategy', sql.Text),
        sql.Column('lookback', sql.BIGINT),
        sql.Column('up', sql.Float),
        sql.Column('down', sql.Float),
        sql.Column('stop', sql.Float),
        sql.Column('take', sql.Float),
        sql.Column('coint', sql.Float),
        sql.Column('statio', sql.Float),
        sql.Column('action', sql.Text),
        sql.Column('zscore', sql.Float),
        sql.Column('per_dev', sql.Float),
        sql.Column('per_dev_c1', sql.Float),
        sql.Column('per_dev_c2', sql.Float),
        sql.Column('l_price', sql.Float),
    )
    meta.create_all(connect)
    return check_table


def create_orders_table(connect):

    meta = sql.MetaData()
    orders_table = sql.Table(
        'orders_log', meta,
        sql.Column('coin1_id', sql.BIGINT),
        sql.Column('coin2_id', sql.BIGINT),
        sql.Column('coin1', sql.Text),
        sql.Column('coin2', sql.Text),
        sql.Column('going_to', sql.Text),
        sql.Column('op_price', sql.Float),
        sql.Column('size1', sql.Float),
        sql.Column('size2', sql.Float),
        sql.Column('op_time', sql.DateTime),
        sql.Column('cl_time', sql.DateTime),
        sql.Column('cl_price', sql.Float),
        sql.Column('result_perc', sql.Float),
        sql.Column('result', sql.Float),
        sql.Column('per_no_commis', sql.Float),
        sql.Column('lookback', sql.BIGINT),
        sql.Column('stop', sql.Float),
        sql.Column('strategy', sql.Text),
        sql.Column('up_from', sql.Float),
        sql.Column('down_to', sql.Float),
        sql.Column('pair', sql.Text),
        sql.Column('up', sql.Float),
        sql.Column('down', sql.Float),
        sql.Column('exchange', sql.Text),
        sql.Column('commis', sql.Float),
        sql.Column('plan_profit', sql.Float),
        sql.Column('c1_op_price', sql.Float),
        sql.Column('c2_op_price', sql.Float),
        sql.Column('c1_cl_price', sql.Float),
        sql.Column('c2_cl_price', sql.Float),
    )
    meta.create_all(connect)
    return orders_table


def update_closedf(connection, coin_id, field_name, q_value):
    idd = int(coin_id)
    close_table = create_close_table(connection)
    if field_name == 'up':
        query = close_table.update().where(close_table.columns.coin1_id == idd).values(up=bindparam('u_name'))
    elif field_name == 'down':
        query = close_table.update().where(close_table.columns.coin1_id == idd).values(down=bindparam('u_name'))
    elif field_name == 'pnl':
        query = close_table.update().where(close_table.columns.coin1_id == idd).values(pnl=bindparam('u_name'))
    elif field_name == 'lookback':
        query = close_table.update().where(close_table.columns.coin1_id == idd).values(lookback=bindparam('u_name'))
    elif field_name == 'stop':
        query = close_table.update().where(close_table.columns.coin1_id == idd).values(stop=bindparam('u_name'))
    else:
        query = close_table.update().where(close_table.columns.coin1_id == idd).values(stop=bindparam('u_name'))

    q_value = str(q_value)
    q_value = q_value.replace(',', '.')
    with connection.connect() as conn:
        if field_name == 'lookback':
            conn.execute(query, u_name=int(q_value))
        else:
            conn.execute(query, u_name=float(q_value))
        # print('изменения внесены')


def update_check_df(connection, pair, field_name, q_value):
    """

    :param connection: Соединение с sqlalchemy core
    :param pair: пара (строка)
    :param field_name: название поля в таблице sql (строка)
    :param q_value: значение параметра, которое надо записать в поле. Должно быть число, но в текстовом формате
    :return:
    """

    check_table = create_check_table(connection)
    if field_name == 'up':
        query = check_table.update().where(check_table.columns.pair == pair).values(up=bindparam('u_name'))
    elif field_name == 'down':
        query = check_table.update().where(check_table.columns.pair == pair).values(down=bindparam('u_name'))
    elif field_name == 'coint':
        query = check_table.update().where(check_table.columns.pair == pair).values(coint=bindparam('u_name'))
    elif field_name == 'zscore':
        query = check_table.update().where(check_table.columns.pair == pair).values(zscore=bindparam('u_name'))
    elif field_name == 'per_dev':
        query = check_table.update().where(check_table.columns.pair == pair).values(per_dev=bindparam('u_name'))
    elif field_name == 'per_dev_c1':
        query = check_table.update().where(check_table.columns.pair == pair).values(per_dev_c1=bindparam('u_name'))
    elif field_name == 'per_dev_c2':
        query = check_table.update().where(check_table.columns.pair == pair).values(per_dev_c2=bindparam('u_name'))
    elif field_name == 'l_price':
        query = check_table.update().where(check_table.columns.pair == pair).values(l_price=bindparam('u_name'))
    elif field_name == 'action':
        query = check_table.update().where(check_table.columns.pair == pair).values(action=bindparam('u_name'))
    elif field_name == 'lookback':
        query = check_table.update().where(check_table.columns.pair == pair).values(lookback=bindparam('u_name'))
    else:
        query = check_table.update().where(check_table.columns.pair == pair).values(statio=bindparam('u_name'))

    with connection.connect() as conn:
        if field_name == 'action':
            conn.execute(query, u_name=q_value)
        elif field_name == 'lookback':
            conn.execute(query, u_name=int(q_value))
        else:
            q_value = q_value.replace(',', '.')
            conn.execute(query, u_name=float(q_value))
        # print('изменения внесены')


def update_orders_df(connection, idd, field_name, q_value):
    """

    :param connection: Соединение с sqlalchemy core
    :param idd: id ордера
    :param field_name: название поля в таблице sql (строка)
    :param q_value: значение параметра, которое надо записать в поле. Должно быть число, но в текстовом формате
    :return:
    """
    idd = int(idd)
    orders_table = create_orders_table(connection)
    # base = orders_table.update().where(orders_table.columns.coin1_id == idd)
    if field_name == 'close_order':
        values = {
                orders_table.columns.op_price: bindparam('u_op_price'),
                orders_table.columns.cl_price: bindparam('u_cl_price'),
                orders_table.columns.cl_time: bindparam('u_cl_time'),
                orders_table.columns.stop: bindparam('u_stop'),
                orders_table.columns.result: bindparam('u_result'),
                orders_table.columns.result_perc: bindparam('u_result_per'),
                orders_table.columns.per_no_commis: bindparam('u_per_no_commis'),
                orders_table.columns.commis: bindparam('u_commis')
                }
        query = orders_table.update().where(orders_table.columns.coin1_id == idd).values(values)
    elif field_name == 'plan_profit':
        query = orders_table.update().where(orders_table.columns.coin1_id == idd).values(plan_profit=bindparam('u_name'))
    else:
        query = orders_table.update().where(orders_table.columns.coin1_id == idd).values(up=bindparam('u_name'))

    with connection.connect() as conn:
        if field_name == 'close_order':
            op_price = float(q_value['op_price'])
            cl_price = float(q_value['cl_price'])
            cl_time = q_value['cl_time']
            stop = float(q_value['stop'])
            result = float(q_value['result'])
            result_perc = float(q_value['result_perc'])
            per_no_commis = float(q_value['per_no_commis'])
            commis = float(q_value['commis'])
            conn.execute(query, u_op_price=op_price, u_cl_price=cl_price, u_cl_time=cl_time, u_stop=stop,
                         u_result=result, u_result_per=result_perc, u_per_no_commis=per_no_commis, u_commis=commis)
        else:
            # q_value = q_value.replace(',', '.')
            conn.execute(query, u_name=float(q_value))
            pass


def delete_row_from_sql(connection, table, q_value):
    if table == 'bin_to_close':
        table_sql = create_close_table(connection)
    else:
        table_sql = create_check_table(connection)
    query = table_sql.delete().where(table_sql.columns.pair == q_value)
    with connection.connect() as conn:
        conn.execute(query)
        # print('изменения внесены')


def sql_table_to_csv(table_name):
    connection = connect_to_sqlalchemy_binance()
    if table_name == 'bin_to_close':
        table_sql = create_close_table(connection)
    elif table_name == 'bin_to_check':
        table_sql = create_check_table(connection)
    elif table_name == 'orders_log':
        table_sql = create_orders_table(connection)
    else:
        table_sql = create_olhc_table(table_name, connection)
    query = table_sql.select()
    with connection.connect() as conn:
        sql_df = pd.read_sql(sql=query, con=conn)
        filepath_close = rf'.\reports\{table_name}.csv'
        sql_df.to_csv(filepath_close, index=False, sep="\t")
        print('Файл таблицы SQL сохранен!')

# #################################################################


# ######################################################################
# Процедуры для расчета эконометрических данных
#
# ######################################################################

def get_delta_adf(data1, data2):
    # Perform ADF test on the closing prices of fetched data
    # на тестах большой разницы между тестом АДФ на обычную цену НЕ показал
    result = stat.OLS(data1, data2).fit()
    ct = ts.adfuller(result.resid, maxlag=1)
    if isinstance(ct, (list, tuple)):
        station = ct[1]
    else:
        station = 1.0
    return station


# рассчитать стационарности ряда
def stationarity(a):
    try:
        a = np.ravel(a)
        stat_res = ts.adfuller(a, maxlag=1)
        if isinstance(stat_res, (list, tuple)):
            stat_coeff = stat_res[1]
        else:
            stat_coeff = 1.0
    except:
        stat_coeff = 1.0
    return stat_coeff


# рассчитать коэф корреляции
def get_corr_coeff(coin1, coin2):
    corr_coeff = np.corrcoef(coin1, coin2)
    return corr_coeff


# рассчитать коэф коинтеграции
def cointegration(a, b):
    coint_coeff = ts.coint(a, b, maxlag=1)
    if isinstance(coint_coeff, (list, tuple)):
        p_value = coint_coeff[1]
    else:
        p_value = 1.0
    return p_value


def cointegration_johansen(a, b):
    # doing an inner join to make sure dates coincide and there are no NaNs
    # inner join requires distinct column names
    main_df = pd.merge(a, b, how='inner', on='startTime')
    # get rid of extra columns but keep the date index
    main_df.drop(labels=['open_x', 'open_y', 'high_x', 'high_y', 'low_x',
                         'low_y', 'time_y'],
                 axis=1,
                 inplace=True)
    main_df.rename(columns={'close_y': 'y', 'close_x': 'x'}, inplace=True)

    # The second and third parameters indicate constant term, with a lag of 1.
    result = coint_johansen(main_df, 0, 1)

    return result


def eg_method(x, y, show_summary=False):
    # use Engle-Granger two-step method to test cointegration
    # the underlying method is straight forward and easy to implement
    # a more important thing is the method is invented by the mentor of my mentor!!!
    # the latest statsmodels package should ve included johansen test which is more common
    # check sm.tsa.var.vecm.coint_johansen
    # the malaise of two-step is the order of the cointegration
    # unlike johansen test, two-step method can only detect the first order
    # check the following material for further details
    # https://warwick.ac.uk/fac/soc/economics/staff/gboero/personal/hand2_cointeg.pdf

    # step 1
    # estimate long run equilibrium
    model1 = stat.OLS(y, stat.add_constant(x)).fit()
    epsilon = model1.resid

    if show_summary:
        print('\nStep 1\n')
        print(model1.summary())

    # check p value of augmented dickey fuller test
    # if p value is no larger than 5%, stationary test is passed
    stat_res = ts.adfuller(epsilon)
    if isinstance(stat_res, (list, tuple)):
        stat_coeff = stat_res[1]
    else:
        stat_coeff = 1.0
    if stat_coeff > 0.05:
        return False, model1

    # take first order difference of X and Y plus the lagged residual from step 1
    x_dif = stat.add_constant(pd.concat([x.diff(), epsilon.shift(1)], axis=1).dropna())
    y_dif = y.diff().dropna()

    # step 2
    # estimate error correction model
    model2 = stat.OLS(y_dif, x_dif).fit()

    if show_summary:
        print('\nStep 2\n')
        print(model2.summary())

    # adjustment coefficient must be negative
    if list(model2.params)[-1] > 0:
        return False, model1
    else:
        return True, model1


def get_statistics(coin1, coin2, coin1_hist, coin2_hist, use_filter=True):

    # Разобраться - почему таблицы разной длинны. Должны быть одинаковой на этапе получения истории
    len1 = len(coin1_hist)
    len2 = len(coin2_hist)
    if len1 != len2:
        if len1 > len2:
            coin1_hist = coin1_hist[:len2]
        else:
            coin2_hist = coin2_hist[:len1]
    if len1 < 50 or len2 < 50:
        # в этом случае не считаем. данных недостаточно
        corr_coeff = 0.0
        coint_coeff = 1.0
        stat_coin1 = 0.0
        stat_coin2 = 0.0
        stat_coin3 = 1.0

    else:

        coin1_hist = prepare_dataframe(df=coin1_hist, timestamp_field="startTime", asc=True)
        coin2_hist = prepare_dataframe(df=coin2_hist, timestamp_field="startTime", asc=True)
        # получим цены раздвижки
        coin3_hist = make_spread_df(coin1_hist, coin2_hist, last_to_end=True)
        # возьмем только цены закрытия
        close_df1 = coin1_hist.close
        close_df2 = coin2_hist.close
        close_df3 = coin3_hist.close

        # рассчитаем стационарность
        stat_coin1 = stationarity(close_df1)
        stat_coin2 = stationarity(close_df2)
        stat_coin3 = stationarity(close_df3)

        # рассчитаем корреляцию и коинтеграцию
        corr_coeff_df = get_corr_coeff(close_df1, close_df2)
        corr_coeff = corr_coeff_df[0][1]
        coint_coeff = cointegration(close_df1, close_df2)
        # coint_coeff_eg = modul.eg_method(close_df1, close_df2, False)

    add_row = False
    if use_filter:
        # было stat_coin3 < 0.015, немного ужесточил
        # снова ужесточил - было stat3<0.01, stats1-2>0.03, cointeg<0.05
        if (stat_coin3 < 0.005) and (stat_coin1 > 0.05) and (stat_coin2 > 0.05) \
                and (coint_coeff < 0.05) and (corr_coeff > 0.8):
            # Пробую отобрать пары стационарные но слабокоинтегрированные
            # if (stat_coin3 < 0.05) and (stat_coin1 > 0.05) and (stat_coin2 > 0.05) \
            # and (coint_coeff < 0.2) and (coint_coeff > 0.05):
            add_row = True
    else:
        add_row = True

    if add_row:
        new_row = pd.DataFrame({
            'pair': [coin1 + '-' + coin2],
            'coin1': [coin1],
            'coin2': [coin2],
            'corr': ["{:.4f}".format(corr_coeff)],
            'coint': ["{:.4f}".format(coint_coeff)],
            'stat1': ["{:.6f}".format(stat_coin1)],
            'stat2': ["{:.6f}".format(stat_coin2)],
            'stat_pair': ["{:.6f}".format(stat_coin3)]
        },
            index=None)
        return new_row
    else:
        return None


def euclidean_distance(coin1_df, coin2_df):
    # проверка данных на длинну, должна быть одинаковой
    len1 = len(coin1_df)
    len2 = len(coin2_df)
    if len1 != len2:
        if len1 > len2:
            coin1_df = coin1_df[:len2]
        else:
            coin2_df = coin2_df[:len1]
    if len1 < 50 or len2 < 50:
        # в этом случае не считаем. данных недостаточно
        distance = 0.0
    else:
        # define two time series as numpy arrays
        ts1 = np.array(coin1_df['close'])
        ts2 = np.array(coin2_df['close'])

        # calculate Euclidean distance
        distance = np.linalg.norm(ts1 - ts2)

    # print the result
    return distance


def get_normalize_pair(df):
    # Normalize time series data
    normalized_df = df.copy()
    check_value = normalized_df.iloc[0]['close']
    multiplier = 1
    if check_value < 10:
        multiplier = get_multiplier(check_value, 1)
    # Compute the first-order difference
    normalized_df['open'] = normalized_df['open'] * multiplier
    normalized_df['high'] = normalized_df['high'] * multiplier
    normalized_df['low'] = normalized_df['low'] * multiplier
    normalized_df['close'] = normalized_df['close'] * multiplier
    # Normalize each column except for 'time' and ''startTime
    for column in df.columns[2:]:
        max_value = df[column].max()
        min_value = df[column].min()
        normalized_df[column] = (df[column] - min_value) / (max_value - min_value)

    return normalized_df


def get_diff_pair(df):
    # Perform differencing to detrend the time series
    detrended_df = df.copy()
    check_value = detrended_df.iloc[0]['close']
    multiplier = 1
    if check_value < 10:
        multiplier = get_multiplier(check_value, 1)
    # Compute the first-order difference
    detrended_df['open'] = detrended_df['open']*multiplier
    detrended_df['high'] = detrended_df['high']*multiplier
    detrended_df['low'] = detrended_df['low']*multiplier
    detrended_df['close'] = detrended_df['close']*multiplier

    detrended_df['open'] = detrended_df['open'].diff()
    detrended_df['high'] = detrended_df['high'].diff()
    detrended_df['low'] = detrended_df['low'].diff()
    detrended_df['close'] = detrended_df['close'].diff()
    # Remove the first row (NaN due to differencing)
    detrended_df = detrended_df.dropna()
    detrended_df = detrended_df.reset_index()
    return detrended_df


def linear_regression_line(close, length):
    if len(close) > length:
        to_cut = len(close) - length
        close = close[to_cut:]
    # Calculate linear regression values
    x = np.arange(len(close))
    slope, intercept = np.polyfit(x, close, deg=1)
    regression_line = slope * x + intercept

    return regression_line


def standard_deviation_channel(df, lookback, num_std_dev):
    close_price = df['close']
    regression_line = linear_regression_line(close_price[-lookback:], lookback)
    std_dev = np.std(close_price[-lookback:])

    upper_channel = regression_line + num_std_dev * std_dev
    lower_channel = regression_line - num_std_dev * std_dev
    df['line_up'] = upper_channel
    df['line_center'] = regression_line
    df['line_down'] = lower_channel
    return df


def rolling_regression_line(close, period):
    regression_line = []
    std_deviation = []
    for i in range(len(close)):
        if i < period:
            regression_line.append(np.nan)
            std_deviation.append(np.nan)
        else:
            x = np.arange(period)
            y = close[i - period + 1:i + 1]
            slope, intercept = np.polyfit(x, y, 1)
            regression_line.append(intercept + slope * (period - 1))
            std_deviation.append(np.std(y))
    return regression_line, std_deviation


def rolling_st_dev_channels(df, lookback, num_std_dev):
    close_price = df['close']
    regression_line, std_dev = rolling_regression_line(close_price, lookback)
    df['line_center'] = regression_line
    df['std_dev'] = std_dev
    df['line_up'] = df['line_center'] + df['std_dev']*num_std_dev
    df['line_down'] = df['line_center'] - df['std_dev']*num_std_dev

    return df


# #######################################################################
def get_multiplier(check_value, n):
    new_value = check_value*10
    if new_value < 10:
        n = get_multiplier(new_value, 10*n)
        return n
    else:
        return n


# ######################################################################
# Процедуры для запроса данных с Бинанс
#
# ######################################################################
def get_coin_min_size(coin):

    min_size = 0.01
    try:
        # get_symbol_info - работает только со спотом. пришлось переписать
        # info = binance_api.get_symbol_info(coin)
        # filters = info["filters"]
        info = binance_api.futures_exchange_info()
        futures_list = info["symbols"]
        f_df = pd.DataFrame(futures_list)
        coin_row = f_df[f_df['pair'] == coin]
        filters = coin_row.iloc[0]['filters']
        for i in filters:
            if i["filterType"] == "LOT_SIZE":
                min_size = i['minQty']
                break
    except Exception as error:
        min_size = 0.01
        print(f'{coin} выдает ошибку в информации по символу - {error}')

    return float(min_size)


def get_last_price(coin):

    try:
        res = binance.fetch_bids_asks([coin])
        df = pd.DataFrame(res)
        df = df.T
    except:
        print('Ошибка при запросе bid/ask')
        time.sleep(1)
        df = get_last_price(coin)
    return df


def get_last_index_price(coin, connection):
    coin_table = create_olhc_table(coin, connection)
    query = coin_table.select()
    with connection.connect() as conn:
        history_df = pd.read_sql(sql=query, con=conn)
        history_df = prepare_dataframe(history_df, timestamp_field="startTime", asc=True)
    last_row = history_df.tail(1)
    return last_row


# получить полный список фьючерсов монет с биржи
def get_all_futures():
    binance.load_markets()
    res = binance.markets
    df = pd.DataFrame(res)
    df = df.T
    df = df[df['quote'] == 'USDT']
    df = df.loc[~df['id'].isin(exception_list)]
    df = df[df['expiry'].isna() == True]
    df = df[df['active'] == True]
    df = df.sort_values('id')
    return df


def get_all_spot_df(source):
    if source == 'ccxt':

        binance_spot = ccxt.binance({
            'enableRateLimit': True,
            'apiKey': key.binanceAPI,
            'secret': key.binanceSecretAPI,
        })
        binance_spot.load_markets()
        res = binance.markets
        df = pd.DataFrame(res)
        df = df.T
        df = df[df['quote'] == 'USDT']
        df = df.loc[~df['id'].isin(exception_list)]
        df = df[df['active'] is True]
        df = df.sort_values('id')
    else:
        exchange_info = binance_api.get_exchange_info()

        df = pd.DataFrame(exchange_info['symbols'])
        df = df[df['quoteAsset'] == 'USDT']
        df = df[df['status'] == 'TRADING']

    return df


def enough_balance(exchange="Binance"):

    try:
        if exchange == "Binance":
            res = binance.fetchBalance()
        elif exchange == "Binance2":
            res = binance2.fetchBalance()
        balance_df = pd.DataFrame(res)
        free_balance = balance_df.loc['free']['USDT']

        return free_balance
    except:
        return 0.0


def get_position(order_id, asset, exchange="Binance"):
    try:
        if exchange == "Binance":
            res = binance.fetch_order(order_id, symbol=asset)
        elif exchange == "Binance2":
            res = binance2.fetch_order(order_id, symbol=asset)
        df = pd.DataFrame([res])
    except:
        df = pd.DataFrame()

    return df
# ====================================================================


# ######################################################################
# Сервисные процедуры
#
# ######################################################################
def pair_to_coins(pair):
    space = pair.find('-')
    coin1 = pair[:space]
    coin2 = pair[space + 1:]
    return coin1, coin2


def read_file(filename) -> bool:
    try:
        with open(file=filename, mode='r') as fh:
            fh.read()
        return True
    except Exception as error:
        print(f'ошибка прочтения файла {filename}, пробуем снова - {error}')
        time.sleep(3)
        read_file(filename=filename)


def prepare_dataframe(df: pd.DataFrame, timestamp_field: str, asc: bool) -> pd.DataFrame:
    """
        Drop duplicate and after sort
    :param df:
    :param timestamp_field: field for check uniq
    :param asc: sorting; if true -> asc, else desc
    :return: sorted dataframe with uniq timestamp
    """
    df.drop_duplicates(subset=[timestamp_field], keep="last", inplace=True)
    df.sort_values(
      by=[timestamp_field],
      ascending=asc,
      inplace=True,
      ignore_index=True,
    )
    return df


def convert_to_tf(df, tf):

    df['startTime'] = pd.to_datetime(df['startTime'])
    df = df.set_index('startTime')
    ohlc = {
        'time': 'last',
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last'
    }
    if tf == 3600:
        tf_str = '1H'
        multipl = 60
    elif tf == 900:
        tf_str = '15min'
        multipl = 15
    else:
        tf_str = '5min'
        multipl = 1

    df = df.resample(tf_str).apply(ohlc)
    df.reset_index(inplace=True)
    if tf != 300:
        this_minute = datetime.datetime.now().minute
        current_candle = int(this_minute / multipl) * multipl
        if current_candle != this_minute:
            hist_last_row = df.iloc[-1]['startTime']
            hist_last_min = pd.to_datetime(hist_last_row).minute
            if hist_last_min == current_candle:
                df = df[:-1]  # бираем последнюю свечу, она еще не закрыта.

    return df


# return intervals with gaps
def get_fetch_intervals(df: pd.DataFrame, date_column_label: str, timeframe: int):
    """
    :param df:
    :param date_column_label: name of timeseries column
    :param timeframe: timeframe in seconds
    :return: [[i1, i2], [i3,i4] ...]
    """

    df["gap"] = df[date_column_label].sort_values().diff() > timeframe*1000
    df['time_from'] = df[date_column_label].shift(1)
    intervals = []
    df_gaps = df[df['gap']==True]

    for d in df_gaps.to_dict(orient="records"):
        intervals.append([d['time_from'], d[date_column_label]])

    df.drop(labels=["gap", "time_from"], axis=1, inplace=True)
    return intervals


def remove_dublicates(df_for_check, df_source):

    if len(df_for_check) > 0:
        values_to_drop = df_source['time'].values
        df_for_check['new'] = ~df_for_check['time'].isin(values_to_drop)
        df_for_check = df_for_check[df_for_check['new'] == True]
        df_for_check.drop(labels=['new'], axis=1, inplace=True)

    return df_for_check
# ======================================================================


# ######################################################################
# Процедуры для работы с историей цен (запрос, добавление, обработка, итд)
# построение спреда
# ######################################################################
def get_binance_historical_klines(asset, tf_str, s_start, end_time, look):

    try:
        res = binance_api.get_historical_klines(
            symbol=asset,
            interval=tf_str,
            start_str=s_start,
            end_str=end_time,
            limit=look,
            klines_type=HistoricalKlinesType.FUTURES
        )
    except:
        time.sleep(1)
        res = get_binance_historical_klines(asset, tf_str, s_start, end_time, look)

    return res


def request_history(asset, tf_str, s_start, lookforward, sql=False, tf=tf_5m):
    """
    :param asset:
    :param tf_str: таймфрейм текстом!!!!
    :param s_start: момент старта в формате timestamp
    :param lookforward: количество свечей в указанном таймфрейме (не в секундах!)
    :param sql: брать данные из sql или нет
    :param tf:
    :return:
    """
    # res = binance.fetch_ohlcv(asset, tf_str, s_start, limit=lookforward)
    rest_candels = lookforward
    df = pd.DataFrame()
    while rest_candels > 0:
        if rest_candels > 1000:
            end_time = s_start + 1000*tf
            s_start = int(s_start*1000)
            end_time = int(end_time*1000)
            look = 1000
        else:
            end_time = s_start + rest_candels*tf
            s_start = int(s_start*1000)
            end_time = int(end_time*1000)
            look = rest_candels
        # TODO - проверить запрос!
        # res = binance_api.futures_historical_klines
        res = get_binance_historical_klines(asset, tf_str, s_start, end_time, look)
        s_start = end_time/1000
        rest_candels = rest_candels - 1000
        part_df = pd.DataFrame(res)
        df = pd.concat([df, part_df], ignore_index=True)

    if len(df) > 0:
        df.drop(df.columns[[5, 6, 7, 8, 9, 10, 11]], axis=1, inplace=True)
        df.columns = ['time', 'open', 'high', 'low', 'close']
        df['startTime'] = df['time'].map(lambda x: datetime.datetime.fromtimestamp(x / 1000))
        if not sql:
            df['startTime'] = df['startTime'].astype(str)
        df['open'] = df['open'].astype(float)
        df['high'] = df['high'].astype(float)
        df['low'] = df['low'].astype(float)
        df['close'] = df['close'].astype(float)
        df = df[['startTime', 'time', 'open', 'high', 'low', 'close']]
        if sql:
            df = prepare_dataframe(df=df, timestamp_field="startTime", asc=True)
        else:
            df = prepare_dataframe(df=df, timestamp_field="startTime", asc=False)
    return df


def add_candels_to_database(coin, df, start, lookforward, connection):
    """
        :param coin: текстовое наименование монеты
        :param df: датафрейм с историческими ценами
        :param start: момент старта в формате timestamp
        :param lookforward: количество свечей до конца нужного периода (в указанном таймфрейме (не в секундах!))
        :param connection: соединение с базой
        :return:
        """

    if len(df) == 0:
        # если массив пустой - значит первичное наполнение
        df = request_history(coin, tf_5m_str, start, lookforward, sql=True)
        with connection.connect() as conn:
            try:
                df.to_sql(name=coin.lower(), con=conn, if_exists='append', index=False)
            except ValueError:
                pass
                # print(f'Запись в базу не получилась - {coin}')  #{error}
    else:
        # наиболее вероятно - что нет последних данных
        last_candle_time = df.tail(1).iloc[0]['time']/1000
        first_candle_time = df.head(1).iloc[0]['time']/1000
        end_time = start + (lookforward * tf_5m - tf_5m)
        # end_time_raw = start + (lookforward * tf_5m - tf_5m)
        # end_time_pd = pd.to_datetime(end_time_raw, unit='s').tz_localize('UTC').tz_convert('Europe/Moscow')
        # end_time = pd.to_datetime(end_time_pd).ceil('5T').timestamp()
        lookforward = int((end_time - last_candle_time)/tf_5m)
        if lookforward < 0:
            lookforward = 0
        lookbackward = int((first_candle_time - start) / tf_5m)
        if lookforward > 0:
            t_start = last_candle_time+tf_5m
            df_temp = request_history(coin, tf_5m_str, t_start, lookforward, sql=True)
            df_temp = remove_dublicates(df_temp, df)
            with connection.connect() as conn:
                try:
                    df_temp.to_sql(name=coin.lower(), con=conn, if_exists='append', index=False)
                except:
                    pass
                    # print(f'Запись в базу не получилась - {coin}')  #{error}
            df = pd.concat([df, df_temp], ignore_index=True)
            df = prepare_dataframe(df=df, timestamp_field="startTime", asc=False)
        if lookbackward > 0:
            df_temp = request_history(coin, tf_5m_str, start, lookbackward, sql=True)
            df_temp = remove_dublicates(df_temp, df)
            with connection.connect() as conn:
                try:
                    df_temp.to_sql(name=coin.lower(), con=conn, if_exists='append', index=False)
                except ValueError:
                    pass
                    # print(f'Запись в базу не получилась - {coin}')  #{error}
            df = pd.concat([df, df_temp], ignore_index=True)
            df = prepare_dataframe(df=df, timestamp_field="startTime", asc=False)
        # потом проверяем на пропуски
        intervals = get_fetch_intervals(df=df, date_column_label="time", timeframe=tf_5m)
        # Заполняем пропуски
        if lookforward == 0:
            lookforward = 1000
        for period in intervals:
            s_start = period[0] / 1000
            while s_start <= (period[1] / 1000):
                df_temp = request_history(coin, tf_5m_str, s_start, lookforward, sql=True)
                df_temp = remove_dublicates(df_temp, df)
                with connection.connect() as conn:
                    try:
                        df_temp.to_sql(name=coin.lower(), con=conn, if_exists='append', index=False)
                    except ValueError:
                        pass
                        # print(f'Запись в базу не получилась - {coin}')  #{error}
                df = pd.concat([df, df_temp], ignore_index=True)
                df = prepare_dataframe(df=df, timestamp_field="startTime", asc=False)
                s_start = s_start + 1000*tf_5m
    return df


def make_spread_df(df_coin1, df_coin2, last_to_end=False, tf=tf_5m):
    """
    :param df_coin1: датафреймы, полученные из get_history_price
    :param df_coin2: датафреймы, полученные из get_history_price
    :param last_to_end: порядок сортировки. при True - последние данные будут в конце, и наоборот
    :param tf:
    :return: датафрейм отношения df_coin1/df_coin2
    """
    # теперь сделать итоговую таблицу
    main_df = pd.merge(df_coin1, df_coin2, how='inner', on='startTime')
    main_df['open'] = main_df['open_x']/main_df['open_y']
    # main_df['high'] = main_df['high_x'] / main_df['low_y']
    # main_df['low'] = main_df['low_x'] / main_df['high_y']
    main_df['high'] = main_df['high_x'] / main_df['close_y']
    main_df['low'] = main_df['low_x'] / main_df['close_y']
    main_df['close'] = main_df['close_x'] / main_df['close_y']
    if 'volume_x' in main_df.columns:
        main_df.drop(labels=['open_x', 'open_y', 'high_x', 'high_y', 'low_x',
                             'low_y', 'close_x', 'close_y', 'volume_x', 'volume_y', 'time_y'],
                     axis=1,
                     inplace=True)
    else:
        main_df.drop(labels=['open_x', 'open_y', 'high_x', 'high_y', 'low_x',
                             'low_y', 'close_x', 'close_y', 'time_y'],
                     axis=1,
                     inplace=True)

    if 'time_x' in main_df.columns:
        main_df = main_df.rename(columns={"time_x": "time"})

    # main_df.index = pd.DatetimeIndex(main_df['startTime'])
    main_df.sort_values(
        by='startTime',
        ascending=last_to_end,
        inplace=True,
        ignore_index=True,
    )

    if tf != tf_5m:
        main_df = convert_to_tf(main_df, tf)

    return main_df


def get_history_price(asset, start, end, tf=tf_5m):
    """
    :param asset:
    :param start:  момент с которого нужны данные в формате timestamp
    :param end:  момент окончания в формате timestamp
    :param tf: количество секунд в нужном таймфрейме
    :return:
    """
    # на данный момент возвращать только 5-м таймфрейм
    if tf == tf_1m:
        tf_str = tf_1m_str
    else:
        tf_str = tf_5m_str

    s_start = start
    filename = f"{asset}.csv"
    filepath = Path("files", filename)
    history_df = pd.DataFrame(columns=['startTime', 'time', 'open', 'high', 'low', 'close'])
    concat_needed = True
    if path.exists(filepath):
        try:
            history_df = pd.read_csv(filepath, sep="\t")
            if isinstance(history_df, pd.DataFrame) and len(history_df) > 0:
                history_df = prepare_dataframe(df=history_df, timestamp_field="startTime", asc=False)

                # the oldest event
                history_start = history_df["time"].values[-1]/1000
                # the newest event
                history_end = history_df["time"].values[0]/1000
                # ситуация, когда start внутри диапазона из файла
                if history_end > start > history_start:
                    if history_end > end:
                        # идеальная ситуация - все данные уже есть в файле
                        concat_needed = False
                    else:
                        # перезаписываем start, нет необходимости запрашивать данные, которые уже есть
                        s_start = history_end
                # ситуации, когда требуемый диапазон полностью выходит за рамки диапазона файла сделать позже
                # и когда дата старта меньше чем дата старта в файле, тоже позже
        except Exception as error:
            print(f'ошибка загрузки истории {asset} из файла - {error}')

    lookforward = end - s_start
    lookforward = int(lookforward / tf)
    df = history_df
    # запросим недостающие данные
    if concat_needed and lookforward != 0:
        # запрашиваем из базы только, если объединение нужно
        try:
            df = request_history(asset, tf_str, s_start, lookforward, False, tf)
            df = pd.concat([history_df, df], ignore_index=True)

            if isinstance(df, pd.DataFrame) and len(df) > 0:
                df = prepare_dataframe(df=df, timestamp_field="startTime", asc=False)
                df.to_csv(filepath, columns=["startTime", "time", "open", "high", "low", "close"], index=False,
                          sep="\t")
        except Exception as ex:
            print(asset, tf, s_start, end, ex)

    df = df[(df['time'] >= start * 1000) & (df['time'] <= end * 1000)]
    return df


def get_sql_history_price(asset, connection, start, end):
    """
    :param asset:
    :param start:  момент с которого нужны данные в формате timestamp
    :param end:  момент окончания в формате timestamp
    :param connection: соединение с базой sql
    :return:
    """

    # 1. выбираем данные за нужный период из базы
    coin_table = create_olhc_table(asset, connection)
    query = coin_table.select().where(coin_table.columns.time >= start * 1000, coin_table.columns.time <= end * 1000)
    with connection.connect() as conn:
        history_df = pd.read_sql(sql=query, con=conn)
    # 2. смотрим, полные ли они, нужна ли дозагрузка
    lookforward = int((end - start) / tf_5m)
    if len(history_df) < lookforward:
        # 3. если да - дозагружаем
        is_index = asset.find("indx")
        if is_index == -1:
            history_df = add_candels_to_database(asset, history_df, start, lookforward, connection)
        else:
            history_df = add_indx_klines_to_database(asset, history_df, start, end, conn)
    # connection.close()
    # 4. возвращаем результат
    return history_df


def get_last_spread_price(coin1, coin2, connection=None):
    # получим последнюю актуальную цену
    is_index = coin1.find("indx")
    if is_index == -1:
        coin1_df = get_last_price(coin1)
        last_price1 = coin1_df.iloc[0]['bid']
    else:
        coin1_df = get_last_index_price(coin1, connection)
        last_price1 = coin1_df.iloc[0]['close']

    coin2_df = get_last_price(coin2)
    last_price2 = coin2_df.iloc[0]['bid']
    l_price = last_price1 / last_price2
    return last_price1, last_price2, l_price


def get_index_history(indx_name, connection, start, end):
    # 1. выбираем данные за нужный период из базы
    coin_table = create_olhc_table(indx_name, connection)
    query = coin_table.select().where(coin_table.columns.time >= start * 1000, coin_table.columns.time <= end * 1000)
    with connection.connect() as conn:
        history_df = pd.read_sql(sql=query, con=conn)
    # 2. смотрим, полные ли они, нужна ли дозагрузка
    lookforward = int((end - start) / tf_5m)
    if len(history_df) < lookforward:
        # 3. если да - дозагружаем
        temp_df = add_indx_klines_to_database(indx_name, history_df, start, end, conn)
    # connection.close()
    # 4. возвращаем результат

    return temp_df


def add_indx_klines_to_database(indx_name, history_df, start, end, conn):

    # 1. сначала получим спецификацию индекса
    filename = f"{indx_name}.csv"
    filepath = Path("reports", filename)
    final_df = pd.DataFrame()
    if path.exists(filepath):
        # 2. запросим необходимые данные по каждой монете из индекса за нужный период
        index_specs = pd.read_csv(filepath, sep="\t")
        for row in range(len(index_specs)):
            coin = index_specs.iloc[row]['coin']
            weight = index_specs.iloc[row]['weight']
            multipl = index_specs.iloc[row]['multipl']
            coin_df = get_sql_history_price(coin, conn, start, end)
            coin_df['close'] = coin_df['close']*multipl*weight
            coin_df['open'] = coin_df['open']*multipl*weight
            coin_df['high'] = coin_df['high']*multipl*weight
            coin_df['low'] = coin_df['low']*multipl*weight

            # 3. рассчитаем индекс
            if len(final_df) > 0:
                final_df['close'] = final_df['close'] + coin_df['close']
                final_df['open'] = final_df['open'] + coin_df['open']
                final_df['high'] = final_df['high'] + coin_df['high']
                final_df['low'] = final_df['low'] + coin_df['low']
            else:
                final_df = coin_df

    # 4. Добавим данные в базу
    df_temp = remove_dublicates(final_df, history_df)
    try:
        df_temp.to_sql(name=indx_name.lower(), con=conn, if_exists='append', index=False)
    except ValueError:
        pass
        # print(f'Запись в базу не получилась - {indx_name}')  #{error}
    df = pd.concat([history_df, df_temp], ignore_index=True)
    df = prepare_dataframe(df=df, timestamp_field="startTime", asc=False)
    return df


def calc_last_data(connection, coin1, coin2, lookback, tf, sigma=3):

    end_time = datetime.datetime.now().timestamp()
    start_time = datetime.datetime.now().timestamp() - lookback*tf*2
    # df_coin1 = modul.get_history_price(coin1, start_time, end_time, tf)
    # df_coin2 = modul.get_history_price(coin2, start_time, end_time, tf)
    df_coin1 = get_sql_history_price(coin1, connection, start_time, end_time)
    df_coin2 = get_sql_history_price(coin2, connection, start_time, end_time)
    df = make_spread_df(df_coin1, df_coin2, True, tf)
    # if len(df_coin1) > lookback and len(df_coin2) > lookback:
    #     df = zscore_calculating(df, lookback)
    # else:
    #     print(f'недостаточно исторических данных по монете {coin1} или {coin2}')
    #     df['zscore'] = 0.0

    df['bb_up'], df['sma'], df['bb_down'] = talib.BBANDS(df.close, lookback, sigma, sigma, 0)

    return df
# ===================================================================


def find_lost_trades(connection, use_sql=True, exchange="Binance"):
    # получим список открытых позиций
    # open_orders = binance.fetch_open_orders()
    if exchange == 'Binance2':
        exch_api = binance2
    else:
        exch_api = binance
    try:
        open_orders = exch_api.fetch_positions()
    except:
        open_orders = [{'symbol': '', 'contracts': 0.0,}]
    pos_df = pd.DataFrame(open_orders)
    pos_df = pos_df[pos_df.contracts != 0.0]
    if len(pos_df) > 0:
        pos_df['coin'] = pos_df['symbol'].str.replace("/", "")
        pos_df['coin'] = pos_df['coin'].str.replace(":USDT", "")
        # посмотрим список позиций, которые должны быть открыты
        if use_sql:
            close_table = create_close_table(connection)
            query = close_table.select().where(close_table.columns.exchange == exchange)
            with connection.connect() as conn:
                close_df = pd.read_sql(sql=query, con=conn)
        else:
            filepath = r'.\reports\bin_to_close.csv'
            if path.exists(filepath):
                close_df = pd.read_csv(filepath, sep="\t")

            else:
                print("не найден файл для закрытия позиций")
                return pd.DataFrame()

        pair_df = close_df[close_df['size2'].isna() == False]
        single_df = close_df[close_df['size2'].isna()]
        new_pair = pd.DataFrame()
        new_single = pd.DataFrame()
        if len(pair_df) > 0:
            pair_df['new1'] = np.where(pair_df.going_to == 'DOWN', -pair_df['size1'], pair_df['size1'])
            pair_df['new2'] = np.where(pair_df.going_to == 'UP', -pair_df['size2'], pair_df['size2'])
            df1 = pair_df[['coin1', 'new1']]
            df2 = pair_df[['coin2', 'new2']]
            pos_df['real_size'] = np.where(pos_df.side == 'short',
                                           -pos_df['contracts'],
                                           pos_df['contracts'])
            df3 = pos_df[['coin', 'real_size']]

            df1.rename(columns={'coin1': 'coin', 'new1': 'size'}, inplace=True)
            df2.rename(columns={'coin2': 'coin', 'new2': 'size'}, inplace=True)

            new_df = pd.concat([df1, df2], ignore_index=True)
            new_df = new_df.groupby(by=['coin'], as_index=False).sum()
            # new_df = pd.merge(new_df, df2, how='outer', on='coin', left_index=False, right_index=False)
            new_pair = pd.concat([new_df, df3], ignore_index=True)
            new_pair = new_pair.groupby(by=['coin'], as_index=False).sum()
            new_pair['diff'] = new_pair['real_size']-new_pair['size']
            new_pair = new_pair[new_pair['diff'] != 0.0]

        if len(single_df) > 0:
            single_df['new1'] = np.where(single_df.going_to == 'DOWN', -single_df['size1'], single_df['size1'])
            # pair_df['new2'] = np.where(pair_df.going_to == 'UP', -pair_df['size2'], pair_df['size2'])
            df1 = single_df[['coin1', 'new1']]
            # df2 = pair_df[['coin2', 'new2']]
            pos_df['real_size'] = np.where(pos_df.side == 'short',
                                           -pos_df['contracts'],
                                           pos_df['contracts'])
            df3 = pos_df[['coin', 'real_size']]

            df1.rename(columns={'coin1': 'coin', 'new1': 'size'}, inplace=True)
            # df2.rename(columns={'coin2': 'coin', 'new2': 'size'}, inplace=True)

            # new_df = pd.concat([df1, df2], ignore_index=True)
            df1 = df1.groupby(by=['coin'], as_index=False).sum()
            # new_df = pd.merge(new_df, df2, how='outer', on='coin', left_index=False, right_index=False)
            new_single = pd.concat([df1, df3], ignore_index=True)
            new_single = new_single.groupby(by=['coin'], as_index=False).sum()
            new_single['diff'] = new_single['real_size']-new_single['size']
            new_single = new_single[new_single['diff'] != 0.0]

        if len(pair_df) == 0 and len(single_df) == 0:
            new_df2 = pd.DataFrame()
            new_df2['coin'] = pos_df['coin']
            new_df2['size'] = pos_df['contracts']
            new_df2['diff'] = pos_df['contracts']
            return new_df2
        else:
            new_df2 = pd.concat([new_pair, new_single], ignore_index=True)
            return new_df2
    else:
        return pd.DataFrame(columns=["coin", "size", "diff"])


def calculate_historical_profit(hist_df, pair, strategy, sma, up, down):
    """

    :param hist_df:
    :param pair: пара текстом
    :param strategy: название стратегии
    :param sma: период SMA
    :param up: уровень верхней зоны
    :param down: уровень нижней зоны
    :return: строка датафрейма с результатами
    """
    hist_df = ind.zscore_calculating(hist_df, sma)
    total, total_per, per_no_commis = 0, 0, 0

    if strategy == 'zscore':
        hist_df['zsc_shift'] = hist_df.shift(periods=1)['zscore']
        # находим пересечения 0 и уровней Up/down, в местах пересечения ставим метки открытия/закрытия
        hist_df['going_to'] = np.where(((hist_df['zscore'] > 0) & (hist_df['zsc_shift'] <= 0)) |
                                       ((hist_df['zscore'] < 0) & (hist_df['zsc_shift'] >= 0)),
                                       'zero',
                                       hist_df['zsc_shift'])
        hist_df['going_to'] = np.where(((hist_df['zscore'] > up) & (hist_df['zsc_shift'] <= up)),
                                       'DOWN',
                                       hist_df['going_to'])
        hist_df['going_to'] = np.where(((hist_df['zscore'] < down) & (hist_df['zsc_shift'] >= down)),
                                       'UP',
                                       hist_df['going_to'])

        # остальные строки удаляем, как не нужные, сдвигаем дф еще раз
        hist_df = hist_df[
            (hist_df.going_to == 'zero') | (hist_df.going_to == 'UP') | (hist_df.going_to == 'DOWN')]
        hist_df['cross_shift'] = hist_df.shift(periods=1)['going_to']

        # оставляем строки, где сигнал изменился (например с UP на zero)
        hist_df = hist_df[hist_df.cross_shift != hist_df.going_to]

        # высчитываем разницу от открытия до закрытия, сумму и процент
        hist_df['close_shift'] = hist_df.shift(periods=-1)['close']
        hist_df['result'] = np.where(hist_df.going_to == 'UP',
                                     round(hist_df['close_shift'] - hist_df['close'], 6),
                                     hist_df['cross_shift'])
        hist_df['result'] = np.where(hist_df.going_to == 'DOWN',
                                     round(hist_df['close'] - hist_df['close_shift'], 6),
                                     hist_df['result'])
        hist_df = hist_df[hist_df.going_to != 'zero']
        hist_df['result_per'] = (hist_df['result'] / hist_df['close'] * 100)
        hist_df['result_per_no'] = hist_df['result_per'] - 0.16
        total = hist_df['result'].sum()
        total_per = hist_df['result_per'].sum()
        per_no_commis = hist_df['result_per_no'].sum()

    total_row = pd.DataFrame({
        'pair': [pair],
        'sma': [sma],
        'down': [down],
        'up': [up],
        'result': ["{:.6f}".format(total)],
        'result_per': ["{:.3f}".format(total_per)],
        'per_no_commis': ["{:.3f}".format(per_no_commis)]},
        index=None)

    return total_row


def get_open_positions(connection, use_sql=True):
    close_df = pd.DataFrame()
    if use_sql:
        close_table = create_close_table(connection)
        query = close_table.select()
        with connection.connect() as conn:
            close_df = pd.read_sql(sql=query, con=conn)
    else:
        filepath_close = r'.\reports\bin_to_close.csv'
        if path.exists(filepath_close):
            if read_file(filepath_close):
                close_df = pd.read_csv(filepath_close, sep="\t")

    return close_df


def get_selected_pairs(connection):

    check_table = create_check_table(connection)
    query = check_table.select()
    with connection.connect() as conn:
        check_df = pd.read_sql(sql=query, con=conn)

    return check_df


def check_for_touch_bb(df, lookback, sigma):
    """

    :param df: датафрейм пары
    :param lookback: период проверки
    :param sigma: отклонение
    :return: разницу в количестве баров между последним пересечением верхней и нижней линии ББ
    """
    if 'bb_up' and 'sma' and 'bb_down' not in df.columns:
        df['bb_up'], df['sma'], df['bb_down'] = talib.BBANDS(df.close, lookback, sigma, sigma, 0)
    df["signal_up"] = np.where(df["close"] > df["bb_up"], 1, 0)
    df["signal_down"] = np.where(df["close"] < df["bb_down"], 1, 0)
    df["shift_up"] = df.shift(periods=1)['signal_up']
    df["shift_down"] = df.shift(periods=1)['signal_down']
    df['mark_up'] = np.where(df["signal_up"] != df["shift_up"], 1, 0)
    df['mark_down'] = np.where(df["signal_down"] != df["shift_down"], 1, 0)
    df_up = df[df['mark_up'] == 1]
    df_down = df[df['mark_down'] == 1]

    last_row = df.tail(1)
    last_time = last_row.iloc[0]['time']
    if len(df_up) > 0:
        last_up = df_up.tail(1)
        # last_up = last_up.reset_index()
        last_time_up = last_up.iloc[0]['time']
    else:
        last_time_up = last_time - lookback*tf_5m*2
    if len(df_down) > 0:
        last_down = df_down.tail(1)
        # last_down = last_down.reset_index()
        last_time_down = last_down.iloc[0]['time']
    else:
        last_time_down = last_time - lookback*tf_5m*2

    if last_time_up > last_time_down:
        time_round = (last_time_up - last_time_down) / 1000 / tf_5m
        time_to_end = (last_time - last_time_up) / 1000 / tf_5m
        time_to_opposite = (last_time - last_time_down) / 1000 / tf_5m
    elif last_time_up < last_time_down:
        time_round = (last_time_down - last_time_up) / 1000 / tf_5m
        time_to_end = (last_time - last_time_down) / 1000 / tf_5m
        time_to_opposite = (last_time - last_time_up) / 1000 / tf_5m
    else:
        time_round = lookback*2
        time_to_end = lookback*2
        time_to_opposite = lookback*2

    return time_round, time_to_end, time_to_opposite


# ####################################################################
# Блок торговых процедур. Возможно, нужно вынести в отдельный модуль.
#
# ####################################################################
def open_pair_position(connection, coin1, coin2, going_to, amount, lookback, stop=0.0,
                       limit=True, strategy='zscore', up_from=0.0, down_to=0.0,
                       use_sql_for_report=True, exchange='Binance'):

    if going_to == 'UP':
        pos_side1 = 'buy'
        pos_side2 = 'sell'
    else:
        pos_side1 = 'sell'
        pos_side2 = 'buy'

    if limit:
        order_data1 = make_limit_order(coin1, amount, pos_side1, 0.0, exchange)
        order_data2 = make_limit_order(coin2, amount, pos_side2, 0.0, exchange)
        try:
            coin1_id = order_data1.iloc[0]['id']
            coin2_id = order_data2.iloc[0]['id']
            c1_price = order_data1.iloc[0]['price']
            c2_price = order_data2.iloc[0]['price']
            price = c1_price/c2_price
            pos_size1 = order_data1.iloc[0]['p_size']
            pos_size2 = order_data2.iloc[0]['p_size']

        except:
            print(f'ошибка при открытии ордера {coin1}/{coin2}')
            return False
    else:
        # определим минимальный лот для первой и второй ноги
        min_size1 = get_coin_min_size(coin1)
        dec1 = Decimal(str(min_size1)).as_tuple().exponent * (-1)

        min_size2 = get_coin_min_size(coin2)
        dec2 = Decimal(str(min_size2)).as_tuple().exponent * (-1)

        coin1_df = get_last_price(coin1)
        coin2_df = get_last_price(coin2)
        if going_to == 'UP':
            last_price1 = coin1_df.iloc[0]['ask']
            last_price2 = coin2_df.iloc[0]['bid']
        else:
            last_price1 = coin1_df.iloc[0]['bid']
            last_price2 = coin2_df.iloc[0]['ask']

        pos_size1 = round(amount / last_price1, dec1)
        order_data1 = place_market_order(coin1, pos_size1, pos_side1, exchange)
        pos_size2 = round(amount / last_price2, dec2)
        order_data2 = place_market_order(coin2, pos_size2, pos_side2, exchange)
        coin1_id = order_data1['id']
        coin2_id = order_data2['id']
        try:
            c1_price = order_data1['price']
            c2_price = order_data2['price']
            price = c1_price/c2_price
        except Exception as error:
            c1_price = last_price1
            c2_price = last_price2
            price = last_price1/last_price2
            print(f'ошибка расчета цены сделки - {error}')

    # добавляем пару к отслеживанию
    new_row = pd.DataFrame({
        'coin1_id': [coin1_id],
        'coin2_id': [coin2_id],
        'pair': [coin1+"-"+coin2],
        'coin1': [coin1],
        'coin2': [coin2],
        'going_to': [going_to],
        'price': [round(price, 6)],
        'stop': [stop],
        'size1': [pos_size1],
        'size2': [pos_size2],
        'strategy': [strategy],
        'lookback': [lookback],
        'up': [up_from],
        'down': [down_to],
        'exchange': [exchange],
        'c1_op_price': [round(c1_price, 6)],
        'c2_op_price': [round(c2_price, 6)]
    },
        index=None)

    save_to_log(coin1_id, new_row, True, connection, sql=True, exchange=exchange)

    if use_sql_for_report:
        with connection.connect() as conn:
            new_row.to_sql(name='bin_to_close', con=conn, if_exists='append', index=False)
    else:
        filepath_close = r'.\reports\bin_to_close.csv'
        if path.exists(filepath_close):
            # добавим к имеющимся парам для отслеживания новые
            file_close_df = pd.read_csv(filepath_close, sep="\t")
            close_df = pd.concat([file_close_df, new_row], ignore_index=True)
            close_df.to_csv(filepath_close, index=False, sep="\t")


def open_single_position(connection, coin1, going_to, amount, lookback, stop=0.0,
                       limit=True, strategy='pp_supertrend', use_sql_for_report=True, exchange='Binance'):

    if going_to == 'UP':
        pos_side1 = 'buy'
    else:
        pos_side1 = 'sell'
    if limit:
        order_data1 = make_limit_order(coin1, amount, pos_side1, 0.0, exchange)

        try:
            coin1_id = order_data1.iloc[0]['id']
            c1_price = order_data1.iloc[0]['price']
            pos_size1 = order_data1.iloc[0]['p_size']

        except:
            print(f'ошибка при открытии ордера {coin1}')
            return False
    else:
        # определим минимальный лот
        min_size1 = get_coin_min_size(coin1)
        dec1 = Decimal(str(min_size1)).as_tuple().exponent * (-1)

        coin1_df = get_last_price(coin1)
        if going_to == 'UP':
            last_price1 = coin1_df.iloc[0]['ask']
        else:
            last_price1 = coin1_df.iloc[0]['bid']

        pos_size1 = round(amount / last_price1, dec1)
        order_data1 = place_market_order(coin1, pos_size1, pos_side1, exchange)
        coin1_id = order_data1['id']
        try:
            c1_price = order_data1['price']
        except Exception as error:
            c1_price = last_price1
            print(f'ошибка расчета цены сделки - {error}')

    # добавляем пару к отслеживанию
    new_row = pd.DataFrame({
        'coin1_id': [coin1_id],
        'pair': [coin1],
        'coin1': [coin1],
        'going_to': [going_to],
        'price': [round(c1_price, 6)],
        'stop': [stop],
        'size1': [pos_size1],
        'strategy': [strategy],
        'lookback': [lookback],
        'exchange': [exchange],
        'c1_op_price': [round(c1_price, 6)]
    },
        index=None)

    save_to_log(coin1_id, new_row, True, connection, sql=True, exchange=exchange)

    if use_sql_for_report:
        with connection.connect() as conn:
            new_row.to_sql(name='bin_to_close', con=conn, if_exists='append', index=False)
    else:
        filepath_close = r'.\reports\bin_to_close.csv'
        if path.exists(filepath_close):
            # добавим к имеющимся парам для отслеживания новые
            file_close_df = pd.read_csv(filepath_close, sep="\t")
            close_df = pd.concat([file_close_df, new_row], ignore_index=True)
            close_df.to_csv(filepath_close, index=False, sep="\t")


def close_position(order_id, coin, size=0.0, limit=False, exchange="Binance"):
    pos_df = get_position(order_id, coin, exchange)
    if len(pos_df) > 0:
        l_side = pos_df.iloc[0]['side']
        if size == 0.0:
            l_size = pos_df.iloc[0]['amount']
        else:
            l_size = size

        if l_side == 'buy':
            new_side = 'sell'
        else:
            new_side = 'buy'

        # теперь делаем обратный ордер
        if limit:
            result = make_limit_order(coin, 0.0, new_side, l_size, exchange)
        else:
            result = place_market_order(coin, l_size, new_side, exchange)
        return result
    else:
        print(f'не закрыта позиция по {coin}! Срочно закрыть вручную!!!')


def close_pair_position(connection, coin1_id, coin2_id, coin1, coin2, size1, size2, l_price, new_row,
                        limit=True, exchange="Binance"):
    # Close positions
    close_position(int(coin1_id), coin1, size1, limit, exchange)
    close_position(int(coin2_id), coin2, size2, limit, exchange)
    save_close_changes(connection, coin1_id)
    print(f'закрыли позицию по {coin1}/{coin2}, цена={l_price}, в {datetime.datetime.now()}')
    save_to_log(coin1_id, new_row, False, connection, sql=True, exchange=exchange)


def close_single_position(connection, coin1_id, coin1, size1, l_price, new_row, limit=True, exchange="Binance"):
    # Close positions
    close_position(int(coin1_id), coin1, size1, limit, exchange)
    save_close_changes(connection, coin1_id)
    print(f'закрыли позицию по {coin1}, цена={l_price}, в {datetime.datetime.now()}')
    save_to_log(coin1_id, new_row, False, connection, sql=True, exchange=exchange)


def place_market_order(coin, p_size, p_side="buy", exchange="Binance"):
    """

    :param coin: название монеты
    :param p_size: размер позиции в единицах монеты!!! Не Доллары!
    :param p_side: направление ордера текстом
    :param exchange: название биржи или субсчета текстом
    :return:
    """
    if exchange == "Binance":
        try:
            o_result = binance.create_order(
                symbol=f"{coin}",
                side=p_side,
                price=0.0,
                amount=p_size,
                type='market'
            )
            return o_result
        except Exception as e:
            print(f'Error making order request: {e}')
            return None
    elif exchange == "Binance2":
        try:
            o_result = binance2.create_order(
                symbol=f"{coin}",
                side=p_side,
                price=0.0,
                amount=p_size,
                type='market'
            )
            return o_result
        except Exception as e:
            print(f'Error making order request: {e}')
            return None


def save_to_log(idd, row, new, connection, sql=True, exchange="Binance"):

    if sql:
        table_sql = create_orders_table(connection)
        query = table_sql.select()
        with connection.connect() as conn:
            log_df = pd.read_sql(sql=query, con=conn)
    else:
        filepath = r'.\reports\bin_to_log.csv'
        log_df = pd.DataFrame()
        if path.exists(filepath):
            log_df = pd.read_csv(filepath, sep="\t")

    row = row.rename(columns={'price': 'op_price'})
    if new:
        row['op_time'] = datetime.datetime.now()
        if sql:
            with connection.connect() as conn:
                try:
                    row.to_sql(name='orders_log', con=conn, if_exists='append', index=False)
                except ValueError:
                    print('Запись в лог ордеров не получилась!')
        else:
            log_df = pd.concat([log_df, row], ignore_index=True)
            log_df.to_csv(filepath, index=False, sep="\t")
    else:
        deal_df = log_df.loc[log_df['coin1_id'] == idd]
        if len(deal_df) > 0:
            ind_row = deal_df.index[0]
            coin1 = log_df.iloc[ind_row]['coin1']
            coin2 = log_df.iloc[ind_row]['coin2']
            coin_id2 = log_df.iloc[ind_row]['coin2_id']
            going_to = log_df.iloc[ind_row]['going_to']
            stop = row.iloc[0]['stop']

            if pd.isna(coin2):  # this is single position
                res_1 = fetch_closed_order(idd, coin1, exchange)
                c1_cl_price = res_1['cl_price']
                if c1_cl_price != 0.0:
                    cl_price = round(c1_cl_price, 6)
                    cl_time = res_1['cl_time']
                    c1_op_price = res_1['op_price']
                    op_price = round(c1_op_price, 6)
                    com_per = res_1['com_per']
                    profit = res_1['profit']
                    if going_to == 'UP':
                        result_perc = (c1_cl_price - c1_op_price) / c1_op_price * 100
                    else:
                        result_perc = (c1_op_price - c1_cl_price) / c1_op_price * 100
                else:
                    cl_price = row.iloc[0]['cl_price']
                    cl_time = datetime.datetime.now()
                    op_price = log_df.iloc[ind_row]['op_price']
                    com_per = 0.08
                    profit = 0.0
                    if going_to == 'UP':
                        result = cl_price - op_price
                    else:
                        result = op_price - cl_price
                    result_perc = result / op_price * 100

            else:  # this is pair position
                res_1 = fetch_closed_order(idd, coin1, exchange)
                res_2 = fetch_closed_order(coin_id2, coin2, exchange)
                c1_cl_price = res_1['cl_price']
                c2_cl_price = res_2['cl_price']
                if c1_cl_price != 0.0 and c2_cl_price != 0.0:
                    cl_price = round(c1_cl_price/c2_cl_price, 6)
                    cl_time = res_2['cl_time']
                    c1_op_price = res_1['op_price']
                    c2_op_price = res_2['op_price']
                    op_price = round(c1_op_price/c2_op_price, 6)
                    com_per = res_1['com_per'] + res_2['com_per']
                    profit = res_1['profit'] + res_2['profit']
                    if going_to == 'UP':
                        coin1_res_perc = (c1_cl_price - c1_op_price) / c1_op_price * 100
                        coin2_res_perc = (c2_op_price - c2_cl_price) / c2_op_price * 100
                    else:
                        coin1_res_perc = (c1_op_price - c1_cl_price) / c1_op_price * 100
                        coin2_res_perc = (c2_cl_price - c2_op_price) / c2_op_price * 100
                    result_perc = coin1_res_perc + coin2_res_perc
                else:
                    cl_price = row.iloc[0]['cl_price']
                    cl_time = datetime.datetime.now()
                    op_price = log_df.iloc[ind_row]['op_price']
                    com_per = 0.16
                    profit = 0.0
                    if going_to == 'UP':
                        result = cl_price - op_price
                    else:
                        result = op_price - cl_price
                    result_perc = result / op_price * 100

            per_no_commis = result_perc - com_per

            # Теперь добавим данные в лог
            if sql:
                values = {
                    'op_price': op_price,
                    'cl_price': cl_price,
                    'cl_time': cl_time,
                    'stop': stop,
                    'result': round(profit, 3),
                    'result_perc': round(result_perc, 3),
                    'per_no_commis': round(per_no_commis, 3),
                    'commis': round(com_per, 3),
                    'c1_cl_price': round(c1_cl_price, 6),
                    'c2_cl_price': 0.0,
                }

                with connection.connect() as conn:
                    update_orders_df(conn, idd, 'close_order', values)

            else:
                log_df.at[ind_row, 'op_price'] = op_price
                log_df.at[ind_row, 'cl_price'] = cl_price
                log_df.at[ind_row, 'cl_time'] = cl_time
                log_df.at[ind_row, 'stop'] = stop
                log_df.at[ind_row, 'result'] = round(profit, 3)
                log_df.at[ind_row, 'result_perc'] = round(result_perc, 3)
                log_df.at[ind_row, 'per_no_commis'] = round(per_no_commis, 3)
                log_df.at[ind_row, 'commis'] = round(com_per, 3)

                log_df.to_csv(filepath, index=False, sep="\t")


def save_close_changes(connection, coin1_id, use_sql_for_report=True):

    if use_sql_for_report:
        close_table = create_close_table(connection)
        query = close_table.delete().where(close_table.columns.coin1_id == int(coin1_id))
        with connection.connect() as conn:
            conn.execute(query)
    else:
        filepath = r'.\reports\bin_to_close.csv'
        if path.exists(filepath):
            close_df = pd.read_csv(filepath, sep="\t")
            # убираем пару из таблицы
            close_df = close_df[close_df['coin1_id'] != coin1_id]
            close_df.to_csv(filepath, index=False, sep="\t")


# ########################################################################
# блок управления лимитными ордерами
#
# ########################################################################
def make_limit_order(coin, amount, p_side, size=0.0, exchange="Binance"):
    """

    :param coin: название монеты
    :param amount: размер позиции в долларах
    :param size: размер позиции в единицах монеты (ТОЛЬКО ДЛЯ ЗАКРЫТИЯ ПОЗИЦИИ)
    :param p_side: направление позиции
    :param exchange:
    :return:
    """

    # Запросить последнюю цену по бид/аск
    coin_df = get_last_price(coin)
    if p_side == 'buy':
        l_price = coin_df.iloc[0]['bid']
    else:
        l_price = coin_df.iloc[0]['ask']

    if size == 0.0:
        # рассчитать размер позиции
        min_size1 = get_coin_min_size(coin)
        if min_size1 >= 1.0:
            dec = 0
        else:
            dec = Decimal(str(min_size1)).as_tuple().exponent * (-1)
        p_size = round(amount / l_price, dec)
    else:
        # если закрываем позицию, то размер уже известен
        p_size = size

    # разместить ордер через place_limit_order
    res = place_limit_order(coin, l_price, p_size, p_side, exchange)
    if res is not None:
        res_df = pd.DataFrame([res])
        order_id = res_df.iloc[0]['id']
        res_df = manage_limit_order(order_id, coin, p_size, p_side, 1, exchange)
    else:
        res_df = pd.DataFrame()

    res_df['p_size'] = p_size

    return res_df


# открытие нового лимитного ордера
def place_limit_order(coin, p_price, p_size, p_side="buy", exchange="Binance"):

    if exchange == "Binance":
        try:
            o_result = binance.create_limit_order(
                symbol=f"{coin}",
                side=p_side,
                amount=p_size,
                price=p_price,
                # params=post_only
            )
            return o_result
        except Exception as e:
            print(f'Ошибка при размещении лимитного ордера по {coin} в {p_side}: {e}')
            return None
    elif exchange == "Binance2":
        try:
            o_result = binance2.create_limit_order(
                symbol=f"{coin}",
                side=p_side,
                amount=p_size,
                price=p_price,
                # params=post_only
            )
            return o_result
        except Exception as e:
            print(f'Ошибка при размещении лимитного ордера по {coin} в {p_side}: {e}')
            return None


def edit_limit_order(order_id, coin, p_side, new_size, l_price, exchange="Binance"):
    # if exchange == "Binance":
    #     try:
    #         res_modify = binance.edit_limit_order(
    #             id=order_id,
    #             symbol=coin,
    #             side=p_side,
    #             amount=new_size,
    #             price=l_price
    #         )
    #         if res_modify is not None:
    #             return True, res_modify
    #     except Exception as error:
    #         # скорее всего за время расчетов ордер успел исполниться, ничего не меняем.
    #         print(f'лимитный ордер изменить не получилось - {error}')
    #         return False, False
    # elif exchange == "Binance2":
    #     try:
    #         res_modify = binance2.edit_limit_order(
    #             id=order_id,
    #             symbol=coin,
    #             side=p_side,
    #             amount=new_size,
    #             price=l_price
    #         )
    #         if res_modify is not None:
    #             return True, res_modify
    #     except Exception as error:
    #         # скорее всего за время расчетов ордер успел исполниться, ничего не меняем.
    #         print(f'лимитный ордер изменить не получилось - {error}')
    #         return False, False

    # изменяем в два этапа - сначала отменяем старый, потом создаем новый.
    res_cancel = cancel_limit_order(order_id, coin, exchange)
    if res_cancel is None:
        # скорее всего за время расчетов ордер успел исполниться, ничего не меняем.
        print(f'лимитный ордер изменить не получилось')
        return False, False
    else:
        res_modify = place_limit_order(coin, l_price, new_size, p_side, exchange)
        return True, res_modify


# отмена открытого ордера
def cancel_limit_order(order_id, coin, exchange="Binance"):

    if exchange == "Binance":
        try:
            o_result = binance.cancel_order(id=order_id, symbol=coin)
            return o_result
        except Exception as e:
            print(f'Ошибка при отмене лимитного ордера по {coin}: {e}')
            return None
    elif exchange == "Binance2":
        try:
            o_result = binance2.cancel_order(id=order_id, symbol=coin)
            return o_result
        except Exception as e:
            print(f'Ошибка при отмене лимитного ордера по {coin}: {e}')
            return None


def fetch_opened_order(order_id, coin, exchange="Binance"):
    time.sleep(1)
    # проверим, исполнен ли ордер
    try:
        if exchange == "Binance":
            res = binance.fetch_order(id=order_id, symbol=coin)
        elif exchange == "Binance2":
            res = binance2.fetch_order(id=order_id, symbol=coin)
    except:
        res = fetch_opened_order(order_id, coin, exchange)
    return res


def fetch_closed_order(order_id, coin, exchange="Binance"):

    # time_since = datetime.datetime.strptime(op_time, '%Y-%m-%d %H:%M:%S.%f').timestamp()*1000-30000
    try:
        request = {'orderId': order_id, }
        if exchange == "Binance":
            trades = binance.fetch_closed_orders(symbol=coin, since=None, limit=None, params=request)
        elif exchange == "Binance2":
            trades = binance2.fetch_closed_orders(symbol=coin, since=None, limit=None, params=request)
        if len(trades) == 2:
            request2 = {'orderId': trades[1]['id'], }
        else:
            trades_df = pd.DataFrame(trades)
            max_row = trades_df['timestamp'].idxmax()
            request2 = {'orderId': trades[max_row]['id'], }
        if exchange == "Binance":
            res1 = binance.fetch_my_trades(symbol=coin, since=None, limit=None, params=request)
            res2 = binance.fetch_my_trades(symbol=coin, since=None, limit=None, params=request2)
        elif exchange == "Binance2":
            res1 = binance2.fetch_my_trades(symbol=coin, since=None, limit=None, params=request)
            res2 = binance2.fetch_my_trades(symbol=coin, since=None, limit=None, params=request2)
        open_df = pd.DataFrame([res1])
        close_df = pd.DataFrame([res2])
        open_df = open_df.T
        close_df = close_df.T
        open_row = open_df.iloc[0][0]
        close_row = close_df.iloc[0][0]
        op_price = float(open_row['price'])
        cl_price = float(close_row['price'])
        cl_time = datetime.datetime.fromtimestamp(close_row['timestamp']/1000)
        if len(open_df) > 1:
            op_comis = 0.0
            op_size = 0.0
            for index in range(len(open_df)):
                op_comis += open_df.iloc[index][0]['fee']['cost']
                op_size += open_df.iloc[index][0]['amount']
        else:
            op_comis = float(open_row['fee']['cost'])
            op_size = open_row['amount']
        if len(close_df) > 1:
            cl_comis = 0.0
            for index in range(len(close_df)):
                cl_comis += close_df.iloc[index][0]['fee']['cost']
        else:
            cl_comis = float(close_row['fee']['cost'])

        open_com_type = open_row['takerOrMaker']
        if open_com_type == 'maker':
            op_com_per = 0.02
        else:
            op_com_per = 0.04
        cl_com_type = close_row['takerOrMaker']
        if cl_com_type == 'maker':
            cl_com_per = 0.02
        else:
            cl_com_per = 0.04
        com_per = op_com_per + cl_com_per
        commis = op_comis + cl_comis
        op_side = open_row['side']
        if op_side == 'buy':
            profit = (cl_price * op_size - op_price * op_size) - commis
        else:
            profit = (op_price * op_size - cl_price * op_size) - commis

        result = {
            'op_price': op_price,
            'cl_price': cl_price,
            'profit': profit,
            'com_per': com_per,
            'commis': round(commis, 2),
            'cl_time': cl_time,
        }
        return result
    except:
        result = {
            'op_price': 0.0,
            'cl_price': 0.0,
            'profit': 0.0,
            'com_per': 0.0,
            'commis': 0.0,
            'cl_time': 0.0,
        }
        return result


# управление ордером до момента полного открытия позиции
def manage_limit_order(order_id, coin, p_size, p_side, count=1, exchange="Binance"):

    # time.sleep(1)
    # проверим, исполнен ли ордер
    res = fetch_opened_order(order_id, coin, exchange)

    result_df = pd.DataFrame([res])
    order_df = pd.DataFrame([res['info']])
    status = order_df.iloc[0]['status']
    full_qty = float(order_df.iloc[0]['origQty'])
    fill_qty = float(order_df.iloc[0]['executedQty'])
    od_price = float(order_df.iloc[0]['price'])
    # проверим исполнение ордера
    if status != 'FILLED':
        # если 'FILLED' - значит ордер исполнен, дальше не смотрим
        # если нет, смотрим нужно ли переместить или изменить размер
        new_size = full_qty - fill_qty
        was_changed = False
        coin_df = get_last_price(coin)
        if p_side == 'buy':
            l_price = coin_df.iloc[0]['bid']
            if l_price > od_price:
                # цена ушла выше, нужно переставить ордер
                was_changed, res_modify = edit_limit_order(order_id, coin, p_side, new_size, l_price, exchange)
        else:
            l_price = coin_df.iloc[0]['ask']
            if l_price < od_price:
                was_changed, res_modify = edit_limit_order(order_id, coin, p_side, new_size, l_price, exchange)

        if was_changed:
            res_df = pd.DataFrame([res_modify])
            order_id = res_df.iloc[0]['id']
            p_size = new_size

        time.sleep(1)
        if count < 4:
            count = count + 1
            res_df = manage_limit_order(order_id, coin, p_size, p_side, count, exchange)
        else:
            cancel_res = cancel_limit_order(order_id, coin, exchange)
            # если отменить не получилось, значит успел исполниться
            if cancel_res is not None:
                res_market = place_market_order(coin, p_size, p_side, exchange)
                res_df = pd.DataFrame([res_market])
            else:
                res_df = result_df
            # res_df = res_df.rename(columns={"orderId": "id"})
        return res_df
    else:
        return result_df
# =========================================================================


if __name__ == '__main__':
    # sql_table_to_csv('bin_to_close')
    # sql_table_to_csv('bin_to_check')
    # sql_table_to_csv('orders_log')
    sql_table_to_csv('AAVEUSDT')
