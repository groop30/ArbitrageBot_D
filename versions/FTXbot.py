import key
import datetime
import requests
import pandas as pd
import talib
from old_FTX.FTXclient import FtxClient
from decimal import Decimal
import plotly.graph_objects as go
import schedule
import time
import scipy.stats as stats
from os import path
from pathlib import Path

base_url = 'https://ftx.com/api/'

ftx_client = FtxClient(
    api_key=key.ftxAPI,
    api_secret=key.ftxSecretAPI)
pd.options.mode.chained_assignment = None


def get_last_price(coin):
    coin_url = f'/markets/{coin}'
    end_url = base_url + coin_url
    coin_info = requests.get(end_url).json()
    df = pd.DataFrame(coin_info)['result']
    return df


# запросить для каждой историю
def get_history_price(asset, start, end, tf):
    # column_names = ["startTime", "time", "open", "high", "low", "close", "value"]
    s_start = start
    filename = f"{asset}.csv"
    filepath = Path("files", filename)
    history_df = pd.DataFrame()
    concat_needed = True
    if path.exists(filepath):
        history_df = pd.read_csv(filepath, sep="\t")
        if isinstance(history_df, pd.DataFrame) and len(history_df) > 0:
            history_df = prepare_dataframe(df=history_df, timestamp_field="startTime", asc=True)
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

    # запросим недостающие данные
    df = history_df
    if concat_needed:
        try:
            # запрашиваем из базы только, если объединение нужно
            res = ftx_client.get_historical_prices(asset, tf, s_start, end)
            df = pd.DataFrame(res)
            df = pd.concat([history_df, df], ignore_index=True)
        except Exception as ex:
            print(asset, tf, s_start, end, ex)

    if isinstance(df, pd.DataFrame) and len(df) > 0:
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
    main_df.drop(labels=['open_x', 'open_y', 'high_x', 'high_y', 'low_x',
                         'low_y', 'close_x', 'close_y', 'volume_x', 'volume_y', 'time_y'],
                 axis=1,
                 inplace=True)
    main_df.index = pd.DatetimeIndex(main_df['startTime'])
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
    df.drop(labels=['z_sma', 'z_std', 'z_diff'], axis=1, inplace=True)
    return df


def make_graph(df):
    fig = go.Figure(data=[go.Candlestick(x=df['startTime'],
                                         open=df['open'],
                                         high=df['high'],
                                         low=df['low'],
                                         close=df['close'])])
    # fig.add_trace(go.Indicator(df["sma"]))
    fig.show()


def save_close_changes(coin1, coin2):
    filepath = r'.\reports\to_close.csv'
    # close_df = pd.DataFrame()

    if path.exists(filepath):
        close_df = pd.read_csv(filepath, sep="\t")
        # убираем пару из таблицы
        close_df['filter'] = close_df['coin1'].astype(str) + '/' + close_df['coin2'].astype(str)
        filter_pair = f'{coin1}/{coin2}'
        close_df = close_df[close_df['filter'] != filter_pair]
        close_df.drop('filter', axis=1, inplace=True)
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


def calc_last_data(coin1, coin2, t_bb, t_zsc):

    tf = 5 * 60
    need_to_cut = False
    # get the biggest period for indicator
    if t_zsc > t_bb:  # для расчета z-score нужно обрезать таблицу
        t_max = t_zsc
    else:
        t_max = t_bb
        need_to_cut = True

    end_time = datetime.datetime.now().timestamp()
    start_time = datetime.datetime.now().timestamp() - t_max*tf
    df = make_spead_df(coin1, coin2, tf, start_time, end_time)
    df['bb_up'], df['sma'], df['bb_down'] = talib.BBANDS(df.close, t_bb, 2, 2, 0)
    # get the biggest period for indicator
    if need_to_cut:  # we
        df = df.tail(t_zsc)
    df['zscore'] = stats.zscore(df.close)
    return df


def check_conditions():
    check_df = pd.DataFrame(columns=["coin1", "coin2", "strategy", "go_up", "go_down"])
    filepath_close = r'.\reports\to_check.csv'
    if path.exists(filepath_close):
        check_df = pd.read_csv(filepath_close, sep="\t")

    # получим данные уже открытых позиций, пока что используем файл to_close
    filepath_close = r'.\reports\to_close.csv'
    to_close_df = pd.read_csv(filepath_close, sep="\t")

    ########################################
    # проверяем каждую пару
    for index in range(len(check_df)):
        coin1 = check_df.iloc[index]['coin1']
        coin2 = check_df.iloc[index]['coin2']

        # если уже есть открытые позиции по этой паре, пока что не открываем ничего
        opened_df = to_close_df[(to_close_df['coin1'] == coin1) & (to_close_df['coin2'] == coin2)]
        if len(opened_df) > 0:
            # TODO - прописать варианты усреднения
            continue

        strategy = check_df.iloc[index]['strategy']
        go_up = check_df.iloc[index]['go_up']
        go_down = check_df.iloc[index]['go_down']

        t_zsc = 100
        t_bb = 200
        if strategy == 'zscore100':
            t_zsc = 100
        df = calc_last_data(coin1, coin2, t_bb, t_zsc)

        # рассчитаем остальные данные
        last_row = df.tail(1)
        l_price = round(last_row.iloc[0]['close'], 6)
        l_bb_up = last_row.iloc[0]['bb_up']
        l_bb_down = last_row.iloc[0]['bb_down']
        l_zscore = last_row.iloc[0]['zscore']
        l_time = datetime.datetime.now()
        if strategy == 'zscore100':
            if l_zscore < go_up:
                print(f'{coin1}/{coin2}-открываем позицию zscore100,UP, цена={l_price}, в {l_time}')
                open_pair_position(coin1, coin2, "UP", 20.0)
            elif l_zscore > go_down:
                print(f'{coin1}/{coin2}-открываем позицию zscore100,DOWN, цена={l_price}, в {l_time}')
                open_pair_position(coin1, coin2, "DOWN", 20.0)
        elif strategy == 'price_range':
            if l_price < go_up:
                print(f'{coin1}/{coin2}-открываем позицию price_range,UP, цена={l_price}, в {l_time}')
            elif l_price > go_down:
                print(f'{coin1}/{coin2}-открываем позицию price_range,DOWN, цена={l_price}, в {l_time}')
        elif strategy == 'bb200':
            if l_price < l_bb_down:
                print(f'{coin1}/{coin2}-открываем позицию bb200,UP, цена={l_price}, в {l_time}')
            elif l_price > l_bb_up:
                print(f'{coin1}/{coin2}-открываем позицию bb200,DOWN, цена={l_price}, в {l_time}')


def open_pair_position(coin1, coin2, going_to, amount):

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
        'size2': [pos_size2]
    },
        index=None)

    save_to_log(idd, new_row, True)

    filepath_close = r'.\reports\to_close.csv'
    if path.exists(filepath_close):
        # добавим к имеющимся парам для отслеживания новые
        file_close_df = pd.read_csv(filepath_close, sep="\t")
        close_df = pd.concat([file_close_df, new_row], ignore_index=True)
        close_df.to_csv(filepath_close, index=False, sep="\t")


def open_file_positions():

    open_df = pd.DataFrame(columns=["coin1", "coin2", "going_to", "amount"])
    filepath_open = r'.\reports\to_open.csv'

    if path.exists(filepath_open):
        open_df = pd.read_csv(filepath_open, sep="\t")

        for index in range(len(open_df)):
            coin1 = open_df.iloc[index]['coin1']
            coin2 = open_df.iloc[index]['coin2']
            going_to = open_df.iloc[index]['going_to']
            amount = open_df.iloc[index]['amount']

            open_pair_position(coin1, coin2, going_to, amount)

            # убираем пару из таблицы
            open_df = open_df[(open_df['coin1'] != coin1) & (open_df['coin2'] != coin2)]

    open_df.to_csv(filepath_open, index=False, sep="\t")


# основная процедура
def check_for_close():
    close_df = pd.DataFrame(columns=["coin1", "coin2", "going_to", "price", "size1", "size2"])
    filepath_close = r'.\reports\to_close.csv'
    if path.exists(filepath_close):
        close_df = pd.read_csv(filepath_close, sep="\t")

    ########################################
    # анализируем открытые позиции, ищем ситуации для усреднения или для закрытия
    for index in range(len(close_df)):
        coin1 = close_df.iloc[index]['coin1']
        coin2 = close_df.iloc[index]['coin2']
        going_to = close_df.iloc[index]['going_to']
        idd = close_df.iloc[index]['id']
        size1 = close_df.iloc[index]['size1']
        size2 = close_df.iloc[index]['size2']

        df = calc_last_data(coin1, coin2, 100, 100)
        l_price = get_last_spread_price(coin1, coin2)

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
        if going_to == 'UP':
            if l_price > l_sma:
                # Close positions
                close_position(coin1, size1)
                close_position(coin2, size2)
                save_close_changes(coin1, coin2)
                print(f'закрыли позицию по {coin1}/{coin2}, цена={l_price}, в {datetime.datetime.now()}')
                save_to_log(idd, new_row, False)
        elif going_to == 'DOWN':
            if l_price < l_sma:
                # Close positions
                close_position(coin1, size1)
                close_position(coin2, size2)
                save_close_changes(coin1, coin2)
                print(f'закрыли позицию по {coin1}/{coin2}, цена={l_price}, в {datetime.datetime.now()}')
                save_to_log(idd, new_row, False)
        ########################################
        # блок условий для открытия позиций усреднения
        # v1 - if price move against me on 2%

        # v2 - if price go outside BB-lines again

        # v3 - if price reach out 3 (then 4, then 5) st dev (z-score >3, 4, 5...)

    # Собственно и все))


check_conditions()
# open_file_positions()
schedule.every().minute.do(check_for_close)
schedule.every(25).seconds.do(check_conditions)
while True:
    schedule.run_pending()
    time.sleep(20)


# global_def()
# mpl.plot(df, type='candle')
# TODO - определять минимальный порог для вхождения в сделку (мин профит)
# TODO - сделать стоплосс???
# TODO - выделить параметр стратегий в отдельное поле
# TODO - прописать стратегию ББ
