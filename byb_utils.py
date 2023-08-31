import key
import requests
import ccxt
import time
import pandas as pd
import datetime
from pathlib import Path
from os import path
# from binance.client import Client
# from binance.enums import HistoricalKlinesType
import bybit
import talib
import numpy as np
import sqlalchemy as sql
from pybit import spot, usdt_perpetual
from decimal import Decimal
import bin_utils as modul

# binance_api = Client(
#     api_key=key.binanceAPI,
#     api_secret=key.binanceSecretAPI
# )
bybit_api = bybit.bybit(test=False, api_key=key.bybitAPI, api_secret=key.bybit_secretAPI)
spot_api =spot.HTTP(endpoint="https://api.bybit.com")
futures_api = usdt_perpetual.HTTP(
    endpoint="https://api.bybit.com",
    api_key = key.bybitAPI,
    api_secret = key.bybit_secretAPI,
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
                  'CHRUSDT', 'FLOWUSDT', 'SUSHIUSDT', 'CVCUSDT']

TELEGRAM_TOKEN =key.Black_Bulls_bot_API
TELEGRAM_CHANNEL = key.TELEGRAM_CHANNEL


def send_message_to_telegram(text):
    try:
        requests.get('https://api.telegram.org/bot{}/sendMessage'.format(TELEGRAM_TOKEN), params=dict(
        chat_id='@blackbullssignals', text=text))
    except:
        print(f'Не получилась отправка сообщения в Telegram - {text}')

# ######################################################################
# Процедуры для работы с SQL
#
# ######################################################################

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


# ######################################################################
# Процедуры для запроса данных с Bybit
#
# ######################################################################
def get_coin_min_size(coin):

    min_size = 0.01
    try:
        info = futures_api.query_symbol()
        futures_list = info["result"]
        f_df = pd.DataFrame(futures_list)
        coin_row = f_df[f_df['name'] == coin]
        min_size = coin_row.iloc[0]['price_scale']
    except Exception as error:
        min_size = 0.01
        print(f'{coin} выдает ошибку в информации по символу - {error}')

    return float(min_size)


def get_last_price_byb(coin):

    res = bybit.fetch_bids_asks([coin])
    df = pd.DataFrame(res)
    df = df.T
    return df


def get_last_index_price_byb(coin, connection):
    coin_table = modul.create_olhc_table(coin, connection)
    query = coin_table.select()
    conn = connection.connect()
    history_df = pd.read_sql(sql=query, con=conn)
    history_df = modul.prepare_dataframe(history_df, timestamp_field="startTime", asc=True)
    last_row = history_df.tail(1)
    return last_row


# получить полный список спотовых монет с байбит
def get_all_spot_bybit():
    spots = bybit.fetch_markets(params={"type": "spot"})
    df = pd.DataFrame(spots)
    df = df[df['quote'] == 'USDT']
    df = df.loc[~df['id'].isin(exception_list)]
    df = df[df['active'] == True]
    df = df.sort_values('id')
    return df


# получить полный список фьючерсов монет с биржи
def get_all_futures_bybit():
    futures = bybit.fetch_markets(params={"type": "future"})
    df = pd.DataFrame(futures)
    df = df[df['quote'] == 'USDT']
    df = df.loc[~df['id'].isin(exception_list)]
    df = df[df['active'] == True]
    df = df.sort_values('id')
    # bybit.fetch_tickers(params={"type": "future"})
    return df


def enough_balance_byb():

    try:
        res = futures_api.Wallet.Wallet_getBalance(coin="USDT").result()
        # balance_df = pd.DataFrame(res)
        free_balance = res[0]['result']['USDT']['avaliable_balance']
        # 10 - это мниимум, что бы можно было закрыть встречные позиции
        if free_balance < 10.0:
            return False
        else:
            return True
    except:
        return False


def get_position_byb(order_id, asset):
    try:
        res = bybit.fetch_order(order_id, symbol=asset)
        df = pd.DataFrame([res])
    except:
        df = pd.DataFrame()

    return df
# ====================================================================


# ######################################################################
# Сервисные процедуры
#
# ######################################################################


# ######################################################################
# Процедуры для работы с историей цен (запрос, добавление, обработка, итд)
# построение спреда
# ######################################################################
def request_history(asset, tf_str, s_start, lookforward, sql=False, tf=tf_5m):
    """
    :param asset:
    :param tf_str: таймфрейм текстом!!!!
    :param s_start: момент старта в формате timestamp
    :param lookforward: количество свечей в указанном таймфрейме (не в секундах!)
    :return:
    """
    rest_candels = lookforward
    df = pd.DataFrame()
    start_time = s_start - 200*tf
    while rest_candels > 0:
        if rest_candels > 200:
            start_time = int(start_time + 200*tf)
        else:
            start_time = int(start_time + rest_candels*tf)

        res = futures_api.query_kline(symbol=asset, interval=5, from_time = start_time)

        rest_candels = rest_candels - 200
        part_df = pd.DataFrame(res['result'])
        df = pd.concat([df, part_df], ignore_index=True)

    if len(df) > 0:
        df.drop(['id', 'symbol', 'interval', 'period', 'start_at', 'volume', 'turnover'], axis=1, inplace=True)
        df = df.rename(columns={'open_time': 'time'})
        df.columns = ['time', 'open', 'high', 'low', 'close']
        df['startTime'] = df['time'].map(lambda x: datetime.datetime.fromtimestamp(x))
        if not sql:
            df['startTime'] = df['startTime'].astype(str)
        df['open'] = df['open'].astype(float)
        df['high'] = df['high'].astype(float)
        df['low'] = df['low'].astype(float)
        df['close'] = df['close'].astype(float)
        df = df[['startTime', 'time', 'open', 'high', 'low', 'close']]
        if sql:
            df = modul.prepare_dataframe(df=df, timestamp_field="startTime", asc=True)
        else:
            df = modul.prepare_dataframe(df=df, timestamp_field="startTime", asc=False)
    return df


def add_candels_to_database(coin, df, start, lookforward, conn):
    """
        :param coin: текстовое наименование монеты
        :param df: датафрейм с историческими ценами
        :param start: момент старта в формате timestamp
        :param lookforward: количество свечей до конца нужного периода (в указанном таймфрейме (не в секундах!))
        :param conn: соединение с базой
        :return:
        """

    if len(df) == 0:
        # если массив пустой - значит первичное наполнение
        df = request_history(coin, tf_5m_str, start, lookforward, sql=True)
        try:
            df.to_sql(name=coin.lower(), con=conn, if_exists='append', index=False)
        except Exception as error:
            print(f'Запись в базу не получилась - {error}')
    else:
        # наиболее вероятно - что нет последних данных
        last_candle_time = df.tail(1).iloc[0]['time']
        first_candle_time = df.head(1).iloc[0]['time']
        end_time = start + (lookforward * tf_5m - tf_5m)
        lookforward = int((end_time - last_candle_time)/tf_5m)
        lookbackward = int((first_candle_time - start) / tf_5m)
        if lookforward > 0:
            t_start = last_candle_time+tf_5m
            df_temp = request_history(coin, tf_5m_str, t_start, lookforward, sql=True)
            df_temp = modul.remove_dublicates(df_temp, df)
            try:
                df_temp.to_sql(name=coin.lower(), con=conn, if_exists='append', index=False)
            except Exception as error:
                print(f'Запись в базу не получилась - {error}')
            df = pd.concat([df, df_temp], ignore_index=True)
            df = modul.prepare_dataframe(df=df, timestamp_field="startTime", asc=False)
        if lookbackward > 0:
            df_temp = request_history(coin, tf_5m_str, start, lookbackward, sql=True)
            df_temp = modul.remove_dublicates(df_temp, df)
            try:
                df_temp.to_sql(name=coin.lower(), con=conn, if_exists='append', index=False)
            except Exception as error:
                print(f'Запись в базу не получилась - {error}')
            df = pd.concat([df, df_temp], ignore_index=True)
            df = modul.prepare_dataframe(df=df, timestamp_field="startTime", asc=False)
        # потом проверяем на пропуски
        intervals = modul.get_fetch_intervals(df=df, date_column_label="time", timeframe=tf_5m)
        # Заполняем пропуски
        for period in intervals:
            s_start = period[1] / 1000
            df_temp = request_history(coin, tf_5m_str, s_start, lookforward, sql=True)
            df_temp = modul.remove_dublicates(df_temp, df)
            try:
                df_temp.to_sql(name=coin.lower(), con=conn, if_exists='append', index=False)
            except Exception as error:
                print(f'Запись в базу не получилась - {error}')
            df = pd.concat([df, df_temp], ignore_index=True)
            df = modul.prepare_dataframe(df=df, timestamp_field="startTime", asc=False)
    return df


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
                history_df = modul.prepare_dataframe(df=history_df, timestamp_field="startTime", asc=False)

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
                df = modul.prepare_dataframe(df=df, timestamp_field="startTime", asc=False)
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
    coin_table = modul.create_olhc_table(asset, connection)
    query = coin_table.select().where(coin_table.columns.time >= start, coin_table.columns.time <= end)
    conn = connection.connect()
    history_df = pd.read_sql(sql=query, con=conn)
    # 2. смотрим, полные ли они, нужна ли дозагрузка
    lookforward = int((end - start) / tf_5m)
    if len(history_df) < lookforward:
        # 3. если да - дозагружаем
        is_index = asset.find("indx")
        if is_index == -1:
            history_df = add_candels_to_database(asset, history_df, start, lookforward, conn)
        else:
            history_df = add_indx_klines_to_database(asset, history_df, start, end, conn)
    # connection.close()
    # 4. возвращаем результат
    return history_df


def get_last_spread_price(coin1, coin2, connection = None):
    # получим последнюю актуальную цену
    is_index = coin1.find("indx")
    if is_index == -1:
        coin1_df = get_last_price_byb(coin1)
        last_price1 = coin1_df.iloc[0]['bid']
    else:
        coin1_df = get_last_index_price_byb(coin1, connection)
        last_price1 = coin1_df.iloc[0]['close']

    coin2_df = get_last_price_byb(coin2)
    last_price2 = coin2_df.iloc[0]['bid']
    l_price = last_price1 / last_price2
    return l_price


def get_index_history(indx_name, connection, start, end):
    # 1. выбираем данные за нужный период из базы
    coin_table = modul.create_olhc_table(indx_name, connection)
    query = coin_table.select().where(coin_table.columns.time >= start * 1000, coin_table.columns.time <= end * 1000)
    conn = connection.connect()
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
    df_temp = modul.remove_dublicates(final_df, history_df)
    try:
        df_temp.to_sql(name=indx_name.lower(), con=conn, if_exists='append', index=False)
    except Exception as error:
        print(f'Запись в базу не получилась - {error}')
    df = pd.concat([history_df, df_temp], ignore_index=True)
    df = modul.prepare_dataframe(df=df, timestamp_field="startTime", asc=False)
    return df


def calc_last_data(connection, coin1, coin2, lookback, tf):

    end_time = datetime.datetime.now().timestamp()
    start_time = datetime.datetime.now().timestamp() - lookback*tf - tf*50
    # df_coin1 = modul.get_history_price(coin1, start_time, end_time, tf)
    # df_coin2 = modul.get_history_price(coin2, start_time, end_time, tf)
    df_coin1 = get_sql_history_price(coin1, connection, start_time, end_time)
    df_coin2 = get_sql_history_price(coin2, connection, start_time, end_time)
    df = modul.make_spread_df(df_coin1, df_coin2, True, tf)
    if len(df_coin1) > lookback and len(df_coin2) > lookback:
        df = modul.zscore_calculating(df, lookback)
    else:
        print(f'недостаточно исторических данных по монете {coin1} или {coin2}')
        df['zscore'] = 0.0

    df['bb_up'], df['sma'], df['bb_down'] = talib.BBANDS(df.close, lookback, 3, 3, 0)

    return df
# ===================================================================


def find_lost_trades(connection, use_sql=True):
    # получим список открытых позиций
    open_orders = bybit.fetch_positions()
    pos_df = pd.DataFrame(open_orders)
    pos_df = pos_df[pos_df.contracts != 0.0]
    if len(pos_df) > 0:
        pos_df['coin'] = pos_df['symbol'].str.replace("/", "")
        # посмотрим список позиций, которые должны быть открыты
        if use_sql:
            close_table = modul.create_close_table(connection)
            query = close_table.select()
            with connection.connect() as conn:
                close_df = pd.read_sql(sql=query, con=conn)
        else:
            filepath = r'.\reports\byb_to_close.csv'
            if path.exists(filepath):
                close_df = pd.read_csv(filepath, sep="\t")

            else:
                print("не найден файл для закрытия позиций")
                return pd.DataFrame()

        close_df['new1'] = np.where(close_df.going_to == 'DOWN', -close_df['size1'], close_df['size1'])
        close_df['new2'] = np.where(close_df.going_to == 'UP', -close_df['size2'], close_df['size2'])
        df1 = close_df[['coin1', 'new1']]
        df2 = close_df[['coin2', 'new2']]
        pos_df['real_size'] = np.where(pos_df.side == 'short',
                                    -pos_df['contracts'],
                                    pos_df['contracts'])
        df3 = pos_df[['coin', 'real_size']]

        df1.rename(columns={'coin1': 'coin', 'new1': 'size'}, inplace=True)
        df2.rename(columns={'coin2': 'coin', 'new2': 'size'}, inplace=True)

        new_df = pd.concat([df1, df2], ignore_index=True)
        new_df = new_df.groupby(by=['coin'], as_index=False).sum()
        # new_df = pd.merge(new_df, df2, how='outer', on='coin', left_index=False, right_index=False)
        new_df2 = pd.concat([new_df, df3], ignore_index=True)
        new_df2 = new_df2.groupby(by=['coin'], as_index=False).sum()
        new_df2['diff'] = new_df2['real_size']-new_df2['size']
        new_df2 = new_df2[new_df2['diff'] != 0.0]
        # print(new_df2)
        return new_df2
    else:
        return pos_df


# TODO дописать процедуру
def find_stop_loss(df, direction):

    # подготовим остальные данные
    last_row = df.tail(1)
    l_sma = last_row.iloc[0]['sma']
    l_price = last_row.iloc[0]['close']

    if direction == 'UP':
        # отберем те свечи, которые за диапазоном
        # df['cross'] = np.where((df['zscore'] > 0) & (df['zsc_shift'] <= 0)),
        #                                'zero',
        #                                df['zsc_shift'])
        # to_sma = l_sma - l_price
        # находим минимальную цену, с момента выхода из диапазона
        stop_price = df['low'].min()
    else:
        # находим максимальную цену, с момента выхода из диапазона
        stop_price = df['high'].max()

    return stop_price


def calculate_historical_profit(hist_df, pair, strategy, sma, up, down):

    hist_df = modul.zscore_calculating(hist_df, sma)
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


# def get_open_positions(connection, use_sql=True):
#     close_df = pd.DataFrame()
#     if use_sql:
#         close_table = modul.create_close_table(connection)
#         query = close_table.select()
#         with connection.connect() as conn:
#             close_df = pd.read_sql(sql=query, con=conn)
#     else:
#         filepath_close = r'.\reports\to_close.csv'
#         if path.exists(filepath_close):
#             if modul.read_file(filepath_close):
#                 close_df = pd.read_csv(filepath_close, sep="\t")
#
#     return close_df


# def get_selected_pairs(connection):
#
#     check_table = create_check_table(connection)
#     query = check_table.select()
#     with connection.connect() as conn:
#         check_df = pd.read_sql(sql=query, con=conn)
#
#     return check_df


def get_max_deviation_from_sma(df, lookback):
    # Принцип расчета:
    # 1. кол. правильных треугольников (изменение уровня дна < 30% от высоты) >30 на 2000
    # 1.1 неправильные считаем отдельно.
    # 2. кол отфильтрованных по времени (если > полураспада - не учитываем в расчетах) <=2
    # 3. кол отфильтрованных по маленькому проценту (< bb-1 (bb-1>0.6%) - не учитываем)
    # 4. средний и макс вынос на правильных треугольниках

    all_deviations = list()
    # добавим данные для расчета
    # df["sma"] = df["close"].rolling(window=lookback, min_periods=1).mean()
    df['bb_up'], df['sma'], df['bb_down'] = talib.BBANDS(df.close, lookback, 1, 1, 0)
    df["signal"] = 0.0
    df["signal"] = np.where(df["close"] > df["sma"], 1.0, 0.0)
    df["position"] = df["signal"].diff()
    df =  df.iloc[lookback:]

    # TODO: DRY
    from_time = df[(df["position"] == 1) | (df["position"] == -1)]
    from_time = from_time.iloc[:-1]
    from_time = from_time.reset_index()
    from_time.drop(["open", "high", "low", "close", "sma", "signal", "position"], axis=1, inplace=True)
    from_time.rename(columns={"startTime": "From"}, inplace=True)
    from_time.rename(columns={"time": "FromTimestm"}, inplace=True)
    # TODO: DRY
    to_time = df[(df["position"] == 1) | (df["position"] == -1)]
    to_time = to_time.iloc[1:]
    to_time = to_time.reset_index()
    to_time.drop(["open", "high", "low", "close", "sma", "signal", "position"], axis=1, inplace=True)
    to_time.rename(columns={"startTime": "To"}, inplace=True)
    to_time.rename(columns={"time": "ToTimestm"}, inplace=True)

    concated_from_to_time = pd.concat([from_time, to_time], axis=1)
    concated_from_to_time.drop(["index"], axis=1, inplace=True)

    # max_deviations_between_crossovers = pd.DataFrame()
    abnormal_count = 0
    low_count = 0
    hl_count = 0
    halflive = lookback/3*2
    for _, row in concated_from_to_time.iterrows():
        # df_slice = df.loc[row["FromTimestm"]:row["ToTimestm"]]
        df_slice = df[(df.time >= row['FromTimestm']) & (df.time <= row['ToTimestm'])]
        if len(df_slice) <= halflive:
            # большей длинны не учитываем, считаем это выносом.
            first_cross = df_slice.iloc[0]['sma']
            bb_up = df_slice.iloc[0]['bb_up']
            bb_down = df_slice.iloc[0]['bb_down']
            # index_of_max_deviation = (df_slice['close'] - df_slice['sma']).abs().idxmax()  # откл от текущего знач sma
            index_of_max_deviation = (df_slice['close'] - first_cross).abs().idxmax() # откл от первого пересечения sma
            row_with_max_deviation = df_slice.loc[[index_of_max_deviation], ['close', 'sma']]  # строка с макс отклонением

            max_dev = row_with_max_deviation
            max_dev["From"] = row["From"]
            max_dev["To"] = row["To"]
            max_dev["len"] = len(df_slice)
            # max_dev["max_deviation"] = (max_dev["close"] - max_dev["sma"])/max_dev["sma"]*100  # %откл от текущего знач sma
            max_dev["max_deviation"] = abs((max_dev["close"] - first_cross) / first_cross * 100) # %откл от первого пересечения sma
            max_deviation = max_dev.iloc[0]["max_deviation"]
            if bb_down < max_dev.iloc[0]["close"]  < bb_up:
                low_count += 1
            else:
                # рассчитаем изменение дна треугольника
                last_cross = df_slice.iloc[len(df_slice)-1]['sma']
                base_div = abs((last_cross - first_cross) / first_cross * 100)
                if base_div <= max_deviation/3:
                    all_deviations.append(max_dev)
                else:
                    abnormal_count += 1
        else:
            hl_count += 1

    if len(all_deviations) > 0:
        full_dev = pd.concat(all_deviations, axis=0, ignore_index=True)
    else:
        full_dev = pd.DataFrame(columns=['close', 'sma', 'From', 'To', 'len', 'max_deviation'])
    # dev_above = full_dev[full_dev["max_deviation"] > 0]
    # dev_below = full_dev[full_dev["max_deviation"] < 0]
    # print(np.std(dev_above["max_deviation"]))
    # print(np.std(dev_below["max_deviation"]))

    return full_dev, low_count, hl_count, abnormal_count

# ####################################################################
# Блок торговых процедур. Возможно, нужно вынести в отдельный модуль.
#
# ####################################################################
def open_pair_position(connection, coin1, coin2, going_to, amount, lookback, stop=0.0,
                       limit=True, strategy='zscore', up_from=0.0, down_to=0.0,
                       use_sql_for_report = True):

    if going_to == 'UP':
        pos_side1 = 'buy'
        pos_side2 = 'sell'
    else:
        pos_side1 = 'sell'
        pos_side2 = 'buy'

    if limit:
        order_data1 = make_limit_order(coin1, amount, pos_side1, 0.0)
        order_data2 = make_limit_order(coin2, amount, pos_side2, 0.0)
        coin1_id = order_data1.iloc[0]['id']
        coin2_id = order_data2.iloc[0]['id']
        price = order_data1.iloc[0]['price']/order_data2.iloc[0]['price']
        pos_size1 = order_data1.iloc[0]['p_size']
        pos_size2 = order_data2.iloc[0]['p_size']
    else:
        # определим минимальный лот для первой и второй ноги
        dec1 = get_coin_min_size(coin1)
        # dec1 = Decimal(str(min_size1)).as_tuple().exponent * (-1)

        dec2 = get_coin_min_size(coin2)
        # dec2 = Decimal(str(min_size2)).as_tuple().exponent * (-1)

        coin1_df = get_last_price_byb(coin1)
        coin2_df = get_last_price_byb(coin2)
        if going_to == 'UP':
            last_price1 = coin1_df.iloc[0]['ask']
            last_price2 = coin2_df.iloc[0]['bid']
        else:
            last_price1 = coin1_df.iloc[0]['bid']
            last_price2 = coin2_df.iloc[0]['ask']

        pos_size1 = round(amount / last_price1, dec1)
        order_data1 = place_market_order(coin1, pos_size1, pos_side1)
        pos_size2 = round(amount / last_price2, dec2)
        order_data2 = place_market_order(coin2, pos_size2, pos_side2)
        coin1_id = order_data1['id']
        coin2_id = order_data2['id']
        try:
            price = order_data1['price']/order_data2['price']
        except Exception as error:
            price = last_price1/last_price2
            print(f'ошибка расчета цены сделки - {error}')

    # добавляем пару к отслеживанию
    new_row = pd.DataFrame({
        'coin1_id': [coin1_id],
        'coin2_id': [coin2_id],
        'pair':[coin1+"-"+coin2],
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
    },
        index=None)

    save_to_log(coin1_id, new_row, True)

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


def close_position(order_id, coin, size=0.0, limit=False):
    pos_df = get_position_byb(order_id, coin)
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
            result = make_limit_order(coin, 0.0, new_side, l_size)
        else:
            result = place_market_order(coin, l_size, new_side)
        return result
    else:
        print(f'не закрыта позиция по {coin}! Срочно закрыть вручную!!!')


def close_pair_position(connection, coin1_id, coin2_id, coin1, coin2, size1, size2, l_price, new_row, limit=True):
    # Close positions
    close_position(coin1_id, coin1, size1, limit)
    close_position(coin2_id, coin2, size2, limit)
    save_close_changes(connection, coin1_id)
    print(f'закрыли позицию по {coin1}/{coin2}, цена={l_price}, в {datetime.datetime.now()}')
    save_to_log(coin1_id, new_row, False)


def place_market_order(coin, p_size, p_side="buy"):
    """

    :param coin: название монеты
    :param p_size: размер позиции в единицах монеты!!! Не Доллары!
    :param p_side: направление ордера текстом
    :return:
    """

    try:
        o_result = bybit.create_order(
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


def save_to_log(idd, row, new):
    filepath = r'.\reports\byb_to_log.csv'
    log_df = pd.DataFrame()
    if path.exists(filepath):
        log_df = pd.read_csv(filepath, sep="\t")

    if new:
        row = row.rename(columns={'price': 'op_price'})
        row['op_time'] = datetime.datetime.now()
        log_df = pd.concat([log_df, row], ignore_index=True)
    else:
        deal_df = log_df.loc[log_df['coin1_id'] == idd]
        if len(deal_df) > 0:
            ind_row = deal_df.index[0]
            cl_price = row.iloc[0]['cl_price']
            stop = row.iloc[0]['stop']
            log_df.at[ind_row, 'cl_price'] = cl_price
            log_df.at[ind_row, 'cl_time'] = datetime.datetime.now()
            log_df.at[ind_row, 'stop'] = stop
            op_price = log_df.iloc[ind_row]['op_price']
            going_to = log_df.iloc[ind_row]['going_to']
            if going_to == 'UP':
                result = cl_price - op_price
            else:
                result = op_price - cl_price
            result_per = result/op_price*100
            log_df.at[ind_row, 'result'] = round(result, 6)
            log_df.at[ind_row, 'result_perc'] = round(result_per, 3)
            per_no_commis = result_per - 0.16
            log_df.at[ind_row, 'per_no_commis'] = round(per_no_commis, 3)

    log_df.to_csv(filepath, index=False, sep="\t")


def save_close_changes(connection, coin1_id, use_sql_for_report=True):

    if use_sql_for_report:
        close_table = modul.create_close_table(connection)
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
def make_limit_order(coin, amount, p_side, size=0.0):
    """

    :param coin: название монеты
    :param amount: размер позиции в долларах
    :param size: размер позиции в единицах монеты (ТОЛЬКО ДЛЯ ЗАКРЫТИЯ ПОЗИЦИИ)
    :param p_side: направление позиции
    :return:
    """

    # Запросить последнюю цену по бид/аск
    coin_df = get_last_price_byb(coin)
    if p_side == 'buy':
        l_price = coin_df.iloc[0]['bid']
    else:
        l_price = coin_df.iloc[0]['ask']

    if size == 0.0:
        # рассчитать размер позиции
        dec = get_coin_min_size(coin)
        # if min_size1 >= 1.0:
        #     dec = 0
        # else:
        #     dec = Decimal(str(min_size1)).as_tuple().exponent * (-1)
        p_size = round(amount / l_price, dec)
    else:
        # если закрываем позицию, то размер уже известен
        p_size = size

    # разместить ордер через place_limit_order
    res = place_limit_order(coin, l_price, p_size, p_side)
    if res is not None:
        res_df = pd.DataFrame([res])
        order_id = res_df.iloc[0]['id']
        res_df = manage_limit_order(order_id, coin, p_size, p_side, 1)
    else:
        res_df = pd.DataFrame()

    res_df['p_size'] = p_size

    return res_df


# открытие нового лимитного ордера
def place_limit_order(coin, p_price, p_size, p_side="buy"):

    # post_only = {'timeInForce': 'PostOnly'}
    try:
        o_result = bybit.create_limit_order(
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


# отмена открытого ордера
def cancel_limit_order(order_id, coin):
    # Place an order
    try:
        o_result = bybit.cancel_order(id=order_id, symbol=coin)
        return o_result
    except Exception as e:
        print(f'Ошибка при отмене лимитного ордера по {coin}: {e}')
        return None


# управление ордером до момента полного открытия позиции
def manage_limit_order(order_id, coin, p_size, p_side, count=1):

    time.sleep(1)
    # проверим, исполнен ли ордер
    res = bybit.fetch_order(id=order_id, symbol=coin)

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
        coin_df = get_last_price_byb(coin)
        if p_side == 'buy':
            l_price = coin_df.iloc[0]['bid']
            if l_price > od_price:
                # цена ушла выше, нужно переставить ордер
                try:
                    res_modify = bybit.edit_limit_order(
                        id=order_id,
                        symbol=coin,
                        side=p_side,
                        amount=new_size,
                        price=l_price
                    )
                    if res_modify is not None:
                        was_changed = True
                except Exception as error:
                    # скорее всего за время расчетов ордер успел исполниться, ничего не меняем.
                    print(f'лимитный ордер изменить не получилось - {error}')
        else:
            l_price = coin_df.iloc[0]['ask']
            if l_price < od_price:
                # цена ушла ниже, нужно переставить ордер
                try:
                    res_modify = bybit.edit_limit_order(
                        id=order_id,
                        symbol=coin,
                        side=p_side,
                        amount=new_size,
                        price=l_price
                    )
                    if res_modify is not None:
                        was_changed = True
                except Exception as error:
                    # скорее всего за время расчетов ордер успел исполниться, ничего не меняем.
                    print(f'лимитный ордер изменить не получилось - {error}')
        if was_changed:
            res_df = pd.DataFrame([res_modify])
            order_id = res_df.iloc[0]['id']
            p_size = new_size

        time.sleep(1)
        if count < 4:
            count = count + 1
            res_df = manage_limit_order(order_id, coin, p_size, p_side, count)
        else:
            cancel_res = cancel_limit_order(order_id, coin)
            # если отменить не получилось, значит успел исполниться
            if cancel_res is not None:
                res_market = place_market_order(coin, p_size, p_side)
                res_df = pd.DataFrame([res_market])
            else:
                res_df = result_df
            # res_df = res_df.rename(columns={"orderId": "id"})
        return res_df
    else:
        return result_df
# =========================================================================