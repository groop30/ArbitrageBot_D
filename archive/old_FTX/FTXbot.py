import key
import datetime
import requests
import pandas as pd
import talib
from FTXclient import FtxClient
from decimal import Decimal
import plotly.graph_objects as go
import schedule
import time
# import scipy.stats as stats
from os import path
from pathlib import Path

base_url = 'https://ftx.com/api/'

ftx_client = FtxClient(
    api_key=key.ftxAPI,
    api_secret=key.ftxSecretAPI)
pd.options.mode.chained_assignment = None
tf_5m = 5 * 60
tf_1h = 60 * 60


def get_last_price(coin):
    coin_url = f'/markets/{coin}'
    end_url = base_url + coin_url
    coin_info = requests.get(end_url).json()
    df = pd.DataFrame(coin_info)['result']
    return df


def read_file(fillename) -> bool:
    try:
        with open(file=fillename, mode='r') as fh:
            fh.read()
        return True
    except:
        time.sleep(3)
        read_file(fillename=fillename)


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
    else:
        tf_str = '5min'

    df = df.resample(tf_str).apply(ohlc)
    return df


def h1_permission(df_1h):

    denied = False
    last_1h = df_1h.tail(2)
    # По тф 1Н смотрим только последнюю полностью сформированную свечу
    high_1h, low_1h, bb_up_1h, bb_down_1h = (
        last_1h[c].to_numpy()[0]
        for c in last_1h
        if c in ["high", "low", "bb_up", "bb_down"]
    )
    # смотрим, что бы последняя свеча была полностью за пределами канала, тогда блокируем действия
    if bb_down_1h > high_1h:
        denied = True
    elif bb_up_1h < low_1h:
        denied = True

    return denied


# запросить для каждой историю
def get_history_price(asset, start, end, tf):

    s_start = start
    filename = f"{asset}.csv"
    filepath = Path("../files", filename)
    history_df = pd.DataFrame()
    concat_needed = True
    if path.exists(filepath):
        if read_file(filepath):
            history_df = pd.read_csv(filepath, sep="\t")
        if isinstance(history_df, pd.DataFrame) and len(history_df) > 0:
            history_df = prepare_dataframe(df=history_df, timestamp_field="startTime", asc=True)
            if tf != tf_5m:
                history_df = convert_to_tf(history_df, tf)
            # the oldest event
            history_start = history_df["time"].values[-1]
            # the newest event
            history_end = history_df["time"].values[0]
            # ситуация, когда start внутри диапазона из файла
            if history_end > start > history_start:
                if history_end > end:
                    # единственная ситуация, когда запрашивать данные от брокера не надо
                    # TODO - make request from db
                    concat_needed = False
                else:
                    # перезаписываем start, нет необходимости запрашивать данные, которые уже есть
                    s_start = history_end
            # ситуации, когда требуемый диапазон полностью выходит за рамки диапазона файла сделать позже
            # и когда дата старта меньше чем дата старта в файле, тоже позже
    df = history_df
    # запросим недостающие данные
    if concat_needed and tf == tf_5m:
        try:
            # запрашиваем из базы только, если объединение нужно
            res = ftx_client.get_historical_prices(asset, tf, s_start, end)
            df = pd.DataFrame(res)
            df = pd.concat([history_df, df], ignore_index=True)
        except Exception as ex:
            print(asset, tf, s_start, end, ex)

    if isinstance(df, pd.DataFrame) and len(df) > 0 and tf == tf_5m:
        df = prepare_dataframe(df=df, timestamp_field="startTime", asc=True)
        df.to_csv(filepath, columns=["startTime", "time", "open", "high", "low", "close"], index=False, sep="\t")

    df = df[(df['time'] >= start*1000) & (df['time'] <= end*1000)]
    return df


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


def place_market_order(coin, p_size, p_side="buy"):
    # Place an order
    try:
        bo_result = ftx_client.place_order(
            market=f"{coin}",
            side=p_side,
            price=0.0,
            size=p_size,
            type='market'
        )
        return bo_result
    except Exception as e:
        print(f'Error making order request: {e}')
        return None


def place_limit_order(coin, p_price, p_size, p_side="buy"):
    # Place an order
    try:
        o_result = ftx_client.place_order(
            market=f"{coin}",
            side=p_side,
            price=p_price,
            size=p_size,
            type='limit'
        )
        return o_result
    except Exception as e:
        print(f'Error making order request: {e}')
        return None


def get_position(asset):
    res = ftx_client.get_position(asset)
    df = pd.DataFrame([res])
    return df


def make_spead_df(coin1, coin2, tf, start_time, end_time):
    # сначала получить историю по монетам
    df_coin1 = get_history_price(coin1, start_time, end_time, tf)
    df_coin2 = get_history_price(coin2, start_time, end_time, tf)
    df_coin1['startTime'] = df_coin1['time'].map(lambda x: datetime.datetime.fromtimestamp(x/1000))
    df_coin2['startTime'] = df_coin2['time'].map(lambda x: datetime.datetime.fromtimestamp(x/1000))
    # теперь сделать итоговую таблицу
    main_df = pd.merge(df_coin1, df_coin2, how='inner', on='startTime')
    main_df['open'] = main_df['open_x']/main_df['open_y']
    main_df['high'] = main_df['high_x'] / main_df['high_y']
    main_df['low'] = main_df['low_x'] / main_df['low_y']
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
    main_df.index = pd.DatetimeIndex(main_df['startTime'])
    # вариант 2, без дропа
    # m_df = df_coin1
    # m_df['open'] = df_coin1['open']/df_coin2['open']
    # m_df['high'] = df_coin1['high'] / df_coin2['high']
    # m_df['low'] = df_coin1['low'] / df_coin2['low']
    # m_df['close'] = df_coin1['close'] / df_coin2['close']
    # m_df.index = pd.DatetimeIndex(m_df['startTime'])
    return main_df


def zscore_calculating(df, lookback):
    # добавляем колонку с sma (A)
    df['z_sma'] = talib.SMA(df.close, lookback)
    # добавляем колонку с отклонением (B)
    df['z_std'] = df.close.rolling(lookback).std()
    # добавляем колонку с разницей sma-цена (A-close)=C
    df['z_diff'] = df['close'] - df['z_sma']
    # рассчитываем zscore как C/B
    df['zscore'] = df['z_diff']/df['z_std']
    df.drop(labels=['z_sma', 'z_std'], axis=1, inplace=True)
    return df


def make_graph(df):
    fig = go.Figure(data=[go.Candlestick(x=df['startTime'],
                                         open=df['open'],
                                         high=df['high'],
                                         low=df['low'],
                                         close=df['close'])])
    # fig.add_trace(go.Indicator(df["sma"]))
    fig.show()


def save_close_changes(idd):

    filepath = r'../reports/to_close.csv'
    if path.exists(filepath):
        close_df = pd.read_csv(filepath, sep="\t")
        # убираем пару из таблицы
        close_df = close_df[close_df['id'] != idd]
        close_df.to_csv(filepath, index=False, sep="\t")


def save_to_log(idd, row, new):
    filepath = r'.\reports\to_log.csv'
    log_df = pd.DataFrame()
    if path.exists(filepath):
        log_df = pd.read_csv(filepath, sep="\t")

    if new:
        row = row.rename(columns={'price': 'op_price'})
        row['op_time'] = datetime.datetime.now()
        log_df = pd.concat([log_df, row], ignore_index=True)
    else:
        deal_df = log_df.loc[log_df['id'] == idd]
        if len(deal_df) > 0:
            ind_row = deal_df.index[0]
            cl_price = row.iloc[0]['cl_price']
            log_df.at[ind_row, 'cl_price'] = cl_price
            log_df.at[ind_row, 'cl_time'] = datetime.datetime.now()
            op_price = log_df.iloc[ind_row]['op_price']
            going_to = log_df.iloc[ind_row]['going_to']
            if going_to == 'UP':
                result = cl_price - op_price
            else:
                result = op_price - cl_price
            result_per = result/op_price*100
            log_df.at[ind_row, 'result'] = round(result, 6)
            log_df.at[ind_row, 'result_perc'] = round(result_per, 3)
            per_no_commis = result_per - 0.272
            log_df.at[ind_row, 'per_no_commis'] = round(per_no_commis, 3)

    log_df.to_csv(filepath, index=False, sep="\t")


def close_position(coin, size=0.0):
    pos_df = get_position(coin)
    # pos_df = pos_df.loc[pos_df['future'] == coin]
    if len(pos_df) > 0:
        l_side = pos_df.iloc[0]['side']
        if size == 0.0:
            l_size = pos_df.iloc[0]['size']
        else:
            l_size = size

        if l_side == 'buy':
            new_side = 'sell'
        else:
            new_side = 'buy'

        # теперь делаем обратный ордер
        result = place_market_order(coin, l_size, new_side)
        return result


def get_last_spread_price(coin1, coin2):
    # получим последнюю актуальную цену
    coin1_df = get_last_price(coin1)
    coin2_df = get_last_price(coin2)
    last_price1 = coin1_df.loc['last']
    last_price2 = coin2_df.loc['last']
    l_price = last_price1 / last_price2
    return l_price


def calc_last_data(coin1, coin2, lookback, tf):

    end_time = datetime.datetime.now().timestamp()
    start_time = datetime.datetime.now().timestamp() - lookback*tf - tf*3
    df = make_spead_df(coin1, coin2, tf, start_time, end_time)
    df = zscore_calculating(df, lookback)
    df['bb_up'], df['sma'], df['bb_down'] = talib.BBANDS(df.close, lookback, 2, 2, 0)

    return df


def check_conditions():
    check_df = pd.DataFrame(columns=["coin1", "coin2", "strategy", "go_up", "go_down"])
    filepath_check = r'../reports/to_check.csv'
    if path.exists(filepath_check):
        if read_file(filepath_check):
            check_df = pd.read_csv(filepath_check, sep="\t")

    # получим данные уже открытых позиций, пока что используем файл to_close
    filepath_close = r'../reports/to_close.csv'
    to_close_df = pd.read_csv(filepath_close, sep="\t")

    ########################################
    # проверяем каждую пару
    for index in range(len(check_df)):

        coin1, coin2, strategy, lookback, go_up, go_down = (
            check_df[c].to_numpy()[index]
            for c in check_df
            if c in ["coin1", "coin2", "strategy", "lookback", "go_up", "go_down"]
        )

        # если разрешение на уровне Н1 не получено - сделок не открываем
        df_1h = calc_last_data(coin1, coin2, lookback, tf_1h)
        h1_denied = h1_permission(df_1h)
        if h1_denied:
            continue

        df = calc_last_data(coin1, coin2, lookback, tf_5m)
        # рассчитаем остальные данные
        l_time = datetime.datetime.now()
        last_row = df.tail(2)
        # получим данные последней строки
        l_price, l_zscore, l_bb_up, l_sma, l_bb_down = (
            last_row[c].to_numpy()[1]
            for c in last_row
            if c in ["close", "zscore", "bb_up", "sma", "bb_down"]
        )
        # получим данные предпоследней строки (последняя сформированная свеча)
        pre_price, pre_zscore, pre_bb_up, pre_sma, pre_bb_down = (
            last_row[c].to_numpy()[0]
            for c in last_row
            if c in ["close", "zscore", "bb_up", "sma", "bb_down"]
        )

        # посмотрим, если ли уже есть открытые позиции
        opened_df = to_close_df[(to_close_df['coin1'] == coin1) & (to_close_df['coin2'] == coin2)]
        ########################################
        # блок условий для открытия позиций усреднения
        if len(opened_df) > 0:
            # усредняем позицию, пока что только если одна позиция.
            if len(opened_df) == 1:
                op_price = opened_df.iloc[0]['price']
                going_to = opened_df.iloc[0]['going_to']
                # значит второй заказ еще не открыт, смотрим условия
                # для zscore 2 вход - при заходе внутрь уровня, если цена от пошла против нас > 1%
                # TODO - если 5(?) предыдущих свечей выходили за диапазон, и цена прошла больше 5%(?)
                # или считать в целевых расстояниях. (ушла против на 1(? 2? 3?) расст до цели- усредняем

                z_condition_1 = False
                z_condition_2 = False
                if strategy == 'zscore':
                    if going_to == 'UP':
                        price_diff = (op_price - l_price)/op_price*100
                        if price_diff > 1:
                            z_condition_1 = True
                        if (pre_zscore < go_up) & (l_zscore > go_up):
                            z_condition_2 = True
                        if z_condition_1 & z_condition_2:
                            # цена ушла ниже, и zsc пересек уровень снизу вверх, пора заходить
                            print(f'{coin1}/{coin2}-усредняем позицию zscore,UP, цена={l_price}, в {l_time}')
                            open_pair_position(coin1, coin2, "UP", 10.0, lookback)
                    else:
                        if going_to == 'DOWN':
                            price_diff = (l_price - op_price) / op_price * 100
                            if price_diff > 1:
                                z_condition_1 = True
                            if (pre_zscore > go_down) & (l_zscore < go_down):
                                z_condition_2 = True
                            if z_condition_1 & z_condition_2:
                                # цена ушла выше, и zsc зашел обратно под уровень, пора заходить
                                print(f'{coin1}/{coin2}-усредняем позицию zscore,DOWN, цена={l_price}, в {l_time}')
                                open_pair_position(coin1, coin2, "DOWN", 10.0, lookback)
                    continue
                else:
                    # TODO - прописать варианты усреднения
                    # для ББ 1 вход - на пересечении линии, 2 - на обратном пересечении (внутрь), если от
                    # первого входа цена ушла > 1%
                    continue
            else:
                continue

        # подготовим данные для дальнейшего анализа
        l_diff = l_price - l_sma
        l_diif_per = l_diff / l_price * 100
        zsc_diff = l_zscore - pre_zscore

        # Если плановый профит меньше 0.6%, то не открываем сделку, т.к. все съест комса и проскальзывание
        if -0.6 < l_diif_per < 0.6:
            continue

        if strategy == 'zscore':
            # для zscore вход при начале уменьшения zsc, если он уже зашел за уровень
            if l_zscore < go_up:
                if zsc_diff > 0.2:
                    # zsc начал увеличиваться, пора заходить
                    print(f'{coin1}/{coin2}-открываем позицию zscore,UP, цена={l_price}, в {l_time}')
                    open_pair_position(coin1, coin2, "UP", 10.0, lookback)
            elif l_zscore > go_down:
                if zsc_diff < -0.2:
                    # zsc начал уменьшаться, пора заходить
                    print(f'{coin1}/{coin2}-открываем позицию zscore,DOWN, цена={l_price}, в {l_time}')
                    open_pair_position(coin1, coin2, "DOWN", 10.0, lookback)
        elif strategy == 'price_range':
            if l_price < go_up:
                print(f'{coin1}/{coin2}-открываем позицию price_range,UP, цена={l_price}, в {l_time}')
            elif l_price > go_down:
                print(f'{coin1}/{coin2}-открываем позицию price_range,DOWN, цена={l_price}, в {l_time}')
        elif strategy == 'bb':
            if l_price < l_bb_down:
                print(f'{coin1}/{coin2}-открываем позицию bb,UP, цена={l_price}, в {l_time}')
                open_pair_position(coin1, coin2, "UP", 10.0, lookback)
            elif l_price > l_bb_up:
                print(f'{coin1}/{coin2}-открываем позицию bb,DOWN, цена={l_price}, в {l_time}')
                open_pair_position(coin1, coin2, "DOWN", 10.0, lookback)


def open_pair_position(coin1, coin2, going_to, amount, lookback):

    coin1_df = get_last_price(coin1)
    coin2_df = get_last_price(coin2)
    if going_to == 'UP':
        pos_side1 = 'buy'
        pos_side2 = 'sell'
        last_price1 = coin1_df.loc['ask']
        last_price2 = coin2_df.loc['bid']
    else:
        pos_side1 = 'sell'
        pos_side2 = 'buy'
        last_price1 = coin1_df.loc['bid']
        last_price2 = coin2_df.loc['ask']

    # определим условия для первой ноги
    min_size1 = coin1_df.loc['minProvideSize']
    dec1 = Decimal(str(min_size1)).as_tuple().exponent * (-1)
    pos_size1 = round(amount / last_price1, dec1)
    order_data1 = place_market_order(coin1, pos_size1, pos_side1)

    # определим условия для первой ноги
    min_size2 = coin2_df.loc['minProvideSize']
    dec2 = Decimal(str(min_size2)).as_tuple().exponent * (-1)
    pos_size2 = round(amount / last_price2, dec2)
    order_data2 = place_market_order(coin2, pos_size2, pos_side2)
    try:
        idd = order_data2['id']
    except Exception as ex:
        idd = order_data1['id']

    price = last_price1/last_price2

    # добавляем пару к отслеживанию
    new_row = pd.DataFrame({
        'id': [idd],
        'coin1': [coin1],
        'coin2': [coin2],
        'going_to': [going_to],
        'price': [round(price, 6)],
        'size1': [pos_size1],
        'size2': [pos_size2],
        'lookback': [lookback]
    },
        index=None)

    save_to_log(idd, new_row, True)

    filepath_close = r'../reports/to_close.csv'
    if path.exists(filepath_close):
        # добавим к имеющимся парам для отслеживания новые
        file_close_df = pd.read_csv(filepath_close, sep="\t")
        close_df = pd.concat([file_close_df, new_row], ignore_index=True)
        close_df.to_csv(filepath_close, index=False, sep="\t")


def open_file_positions():

    open_df = pd.DataFrame(columns=["coin1", "coin2", "going_to", "amount"])
    filepath_open = r'../reports/to_open.csv'

    if path.exists(filepath_open):
        open_df = pd.read_csv(filepath_open, sep="\t")

        for index in range(len(open_df)):
            coin1 = open_df.iloc[index]['coin1']
            coin2 = open_df.iloc[index]['coin2']
            going_to = open_df.iloc[index]['going_to']
            amount = open_df.iloc[index]['amount']
            lookback = open_df.iloc[index]['lookback']
            open_pair_position(coin1, coin2, going_to, amount, lookback)

            # убираем пару из таблицы
            open_df = open_df[(open_df['coin1'] != coin1) & (open_df['coin2'] != coin2)]

    open_df.to_csv(filepath_open, index=False, sep="\t")


# основная процедура
def check_for_close():
    close_df = pd.DataFrame(columns=["coin1", "coin2", "going_to", "price", "size1", "size2", "lookback"])
    filepath_close = r'../reports/to_close.csv'
    if path.exists(filepath_close):
        close_df = pd.read_csv(filepath_close, sep="\t")

    ########################################
    # анализируем открытые позиции, ищем ситуации для усреднения или для закрытия
    for index in range(len(close_df)):
        # получим данные строки
        idd, coin1, coin2, going_to, size1, size2, lookback = (
            close_df[c].to_numpy()[index]
            for c in close_df
            if c in ["id", "coin1", "coin2", "going_to", "size1", "size2", "lookback"]
        )
        df = calc_last_data(coin1, coin2, lookback, tf_5m)
        l_price = get_last_spread_price(coin1, coin2)

        # если разрешение на уровне Н1 не получено - закрываем открытые позиции
        df_1h = calc_last_data(coin1, coin2, lookback, tf_1h)
        h1_denied = h1_permission(df_1h)

        # подготовим остальные данные
        last_row = df.tail(1)
        l_sma = last_row.iloc[0]['sma']

        # создаем строку с данными
        new_row = pd.DataFrame({
            'coin1': [coin1],
            'coin2': [coin2],
            'going_to': [going_to],
            'cl_price': [round(l_price, 6)]
        },
            index=None)
        ########################################
        # блок условий для закрытия позиций
        if h1_denied:
            close_position(coin1, size1)
            close_position(coin2, size2)
            save_close_changes(idd)
            print(f'закрыли позицию по {coin1}/{coin2},(стоп по Н1) цена={l_price}, в {datetime.datetime.now()}')
            save_to_log(idd, new_row, False)

        if going_to == 'UP':
            if l_price > l_sma:
                # Close positions
                close_position(coin1, size1)
                close_position(coin2, size2)
                save_close_changes(idd)
                print(f'закрыли позицию по {coin1}/{coin2}, цена={l_price}, в {datetime.datetime.now()}')
                save_to_log(idd, new_row, False)
        elif going_to == 'DOWN':
            if l_price < l_sma:
                # Close positions
                close_position(coin1, size1)
                close_position(coin2, size2)
                save_close_changes(idd)
                print(f'закрыли позицию по {coin1}/{coin2}, цена={l_price}, в {datetime.datetime.now()}')
                save_to_log(idd, new_row, False)


check_conditions()
# open_file_positions()
schedule.every().minute.do(check_for_close)
schedule.every(25).seconds.do(check_conditions)
while True:
    schedule.run_pending()
    time.sleep(20)

# TODO - проверить на истории перевод в безубыток (цена прошла 50%(?) движения, и развернулась обратно)
# TODO - сделать стоплосс???
# TODO - возможно - отмечать уровни хай/лоу в диапазоне проверки коинт/стац, и их использовать как стопы
#  смотреть расстояние от хая до тейка, и такое же расстояние считать стопом
# TODO - прописать стратегию ББ
# TODO - смотреть на 1h, торговля к средней по 1h, и стоп при выносе.
# TODO - тейк увеличить (до противоположного уровня? до % от уровня?)
