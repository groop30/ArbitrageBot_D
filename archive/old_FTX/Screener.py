import numpy as np
import pandas as pd
# import key
import datetime
from os import path
from statsmodels.tsa.stattools import coint, adfuller
from pathlib import Path
from typing import Optional
from old_FTX.FTXclient import FtxClient

ftx_client = FtxClient(
        api_key='',
        api_secret='')
pd.options.mode.chained_assignment = None

result_df = pd.DataFrame(columns=['coin1', 'coin2', 'corr', 'coint', 'stat1', 'stat2'], index=None)

exception_list = ['AMPL-PERP', 'PRIV-PERP', 'BIT-PERP', 'KSHIB-PERP', 'LUNA2-PERP', 'HOLY-PERP', 'BTT-PERP',
                  'KSOS-PERP', 'SOS-PERP', 'MVDA10-PERP', 'MVDA25-PERP', 'BTT-PERP', 'KBTT-PERP', 'PAXG-PERP',
                  'MVDA25-PERP', 'PAXG-PERP', 'RON-PERP', 'UNISWAP-PERP', 'XAUT-PERP', 'YFII-PERP']


# получить полный список фьючерсов монет с биржи
def get_all_futures():
    res = ftx_client.get_all_futures()
    df = pd.DataFrame(res)
    return df


# запросить для каждой историю
def get_history_price(asset, start, end, tf):
    # column_names = ["startTime", "time", "open", "high", "low", "close", "value"]
    s_start = start
    filename = f"{asset}.csv"
    filepath = Path("../files", filename)
    history_df = pd.DataFrame()
    concat_needed = True
    if path.exists(filepath):
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
                    # TODO - make request from db
                    concat_needed = False
                else:
                    # перезаписываем start, нет необходимости запрашивать данные, которые уже есть
                    s_start = history_end
            # ситуации, когда требуемый диапазон полностью выходит за рамки диапазона файла сделать позже
            # и когда дата старта меньше чем дата старта в файле, тоже позже

    # запросим недостающие данные
    if concat_needed:
        # запрашиваем из базы только, если объединение нужно
        res = ftx_client.get_historical_prices(asset, tf, s_start, end)
        df = pd.DataFrame(res)
        df = pd.concat([history_df, df], ignore_index=True)
        if isinstance(df, pd.DataFrame) and len(df) > 0:
            df = prepare_dataframe(df=df, timestamp_field="startTime", asc=False)
            df.to_csv(filepath, columns=["startTime", "time", "open", "high", "low", "close"], index=False, sep="\t")
    else:
        df = history_df

    df = df[(df['time'] >= start * 1000) & (df['time'] <= end * 1000)]
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


def make_spread_df(df_coin1, df_coin2):

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
    return main_df


# return intervals with gaps
def get_fetch_intervals(df: pd.DataFrame, date_column_label: str, timeframe: int):
    """
    :param df:
    :param date_column_label: name of timeseries column
    :param timeframe: timeframe in seconds
    :return: [[i1, i2], [i3,i4] ...]
    """

    df["gap"] = df[date_column_label].sort_values().diff() > timeframe*1000

    intervals = []

    previous_timestamp: Optional[int] = None
    gap_finded = False
    for d in df.to_dict(orient="records"):
        if previous_timestamp:
            if gap_finded:
                intervals.append([previous_timestamp, d[date_column_label]])
                gap_finded = False
            if d["gap"]:
                gap_finded = True
                previous_timestamp = d[date_column_label]
                continue

        previous_timestamp = d[date_column_label]
        # if not d["gap"]:
        #    continue

    df.drop(labels=["gap"], axis=1, inplace=True)
    return intervals


# рассчитать коэф корреляции
def get_corr_coeff(coin1, coin2):
    corr_coeff = np.corrcoef(coin1, coin2)
    return corr_coeff


# рассчитать стационарности ряда
def stationarity(a):
    a = np.ravel(a)
    stat_coeff = adfuller(a)[1]
    stat_coeff = stat_coeff
    return stat_coeff


# рассчитать коэф коинтеграции
def cointegration(a, b):
    coint_coeff = coint(a, b)[1]
    return coint_coeff


# основная процедура
def get_screening_result():
    global result_df
    global exception_list
    # получим список всех монет
    all_coins = get_all_futures()
    all_coins = all_coins.loc[~all_coins['name'].isin(exception_list)]
    all_coins = all_coins[all_coins.name.str.find('PERP') > -1]
    all_coins = all_coins[all_coins.volumeUsd24h > 100000.0]
    all_futures = all_coins.name.tolist()
    all_futures2 = all_coins.name.tolist()
    corr_dict = dict()
    tf = 5 * 60
    end_time = datetime.datetime.now().timestamp()
    start_time = datetime.datetime.now().timestamp() - 2000 * tf
    for future in all_futures:
        # каждую монету сравним со всеми остальными
        coin1_hist = get_history_price(future, start_time, end_time, tf)
        if len(coin1_hist) > 0:
            print("сравниваем монету " + future)
            for future2 in all_futures2:
                if future2 != future:
                    coin2_hist = get_history_price(future2, start_time, end_time, tf)
                    if len(coin2_hist) > 0:
                        # Разобраться - почему таблицы разной длинны. Должны быть одинаковой на этапе получения истории
                        len1 = len(coin1_hist)
                        len2 = len(coin2_hist)
                        if len1 != len2:
                            if len1 > len2:
                                coin1_hist = coin1_hist[:len2]
                            else:
                                coin2_hist = coin2_hist[:len1]
                        # получим цены раздвижки
                        coin3_hist = make_spread_df(coin1_hist, coin2_hist)

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
                        if (stat_coin3 < 0.0005) and (stat_coin1 > 0.0005) and (stat_coin2 > 0.0005) \
                                and (coint_coeff < 0.05) and (corr_coeff > 0.8):
                            new_row = pd.DataFrame({
                                'coin1': [future],
                                'coin2': [future2],
                                'corr': ["{:.4f}".format(corr_coeff)],
                                'coint': ["{:.4f}".format(coint_coeff)],
                                'stat1': ["{:.6f}".format(stat_coin1)],
                                'stat2': ["{:.6f}".format(stat_coin2)],
                                'stat3': ["{:.6f}".format(stat_coin3)],
                                },
                                index=None)
                            result_df = pd.concat([result_df, new_row], ignore_index=True)
                            print('добавлена строка по паре ' + future + '/' + future2)
                            result_df.to_csv(r'.\reports\total.csv', sep="\t", float_format='%.6f')
                    else:
                        print("по монете "+future2+" данные не получены")
        else:
            print("по монете " + future + " данные не получены")

        all_futures2.remove(future)

    print(corr_dict)


def fill_history_gaps(coin, tf):
    filename = f"{coin}.csv"
    filepath = Path("../files", filename)
    to_add = False
    if path.exists(filepath):
        file_df = pd.read_csv(filepath, sep="\t")
        if isinstance(file_df, pd.DataFrame) and len(file_df) > 0:
            file_df = prepare_dataframe(df=file_df, timestamp_field="startTime", asc=False)
            intervals = get_fetch_intervals(df=file_df, date_column_label="time", timeframe=tf)
            for period in intervals:
                print(f'есть пробелы в {coin}, заполняем')
                s_start = period[1] / 1000
                end = period[0] / 1000
                res = ftx_client.get_historical_prices(coin, tf, s_start, end)
                df = pd.DataFrame(res)
                file_df = pd.concat([file_df, df], ignore_index=True)
                to_add = True

        if isinstance(file_df, pd.DataFrame) and len(file_df) > 0:
            file_df = prepare_dataframe(df=file_df, timestamp_field="startTime", asc=False)
            file_df.to_csv(filepath, columns=["startTime", "time", "open", "high", "low", "close"], index=False,
                           sep="\t")
    return to_add


# проверяет пропуски периодов во всех таблицах данных
def fill_all_gaps(list=None, n=1):
    if list is None:
        all_coins = get_all_futures()
        all_coins = all_coins[all_coins.name.str.find('PERP') > -1]
        list = all_coins.name.tolist()
    new_list = []
    tf = 60 * 5
    for future in list:
        print(f'проверка монеты {future}')
        to_add = fill_history_gaps(future, tf)
        if to_add:
            new_list.append(future)
    if (len(new_list) > 0) & (n < 4):
        n = n + 1
        fill_all_gaps(new_list, n)


def check_parameters_stability(window, steps, shift, file):
    """
    :param window: глубина каждого периода проверки (в свечках)
    :param steps: количество шагов проверки
    :param shift: сдвиг назад при каждой новой проверке (в свечках)
    :param file: какой файл проверять
    :return:
    """
    res_df = pd.DataFrame(columns=['coin1', 'coin2', 'start', 'end', 'corr', 'coint', 'stat1', 'stat2'], index=None)
    tf = 5 * 60
    if file == 1:
        filepath_close = r'../reports/to_check.csv'
    elif file == 2:
        filepath_close = r'../reports/to_check_waiting_list.csv'
    else:
        filepath_close = r'../reports/xxx.csv'
    if path.exists(filepath_close):
        check_df = pd.read_csv(filepath_close, sep="\t")
        for index in range(len(check_df)):
            coin1 = check_df.iloc[index]['coin1']
            coin2 = check_df.iloc[index]['coin2']
            print(f'проверяем пару {coin1}/{coin2}')
            for step in range(steps):
                end_time = datetime.datetime.now().timestamp() - step * shift * tf
                start_time = end_time - window * tf
                coin1_hist = get_history_price(coin1, start_time, end_time, tf)
                coin2_hist = get_history_price(coin2, start_time, end_time, tf)
                if len(coin1_hist) > 0 and len(coin2_hist) > 0:
                    # Разобраться - почему таблицы разной длинны. Должны быть одинаковой на этапе получения истории
                    len1 = len(coin1_hist)
                    len2 = len(coin2_hist)
                    if len1 != len2:
                        if len1 > len2:
                            coin1_hist = coin1_hist[:len2]
                        else:
                            coin2_hist = coin2_hist[:len1]
                    # получим цены раздвижки
                    coin3_hist = make_spread_df(coin1_hist, coin2_hist)

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
                    new_row = pd.DataFrame({
                        'coin1': [coin1],
                        'coin2': [coin2],
                        'start': [datetime.datetime.fromtimestamp(start_time)],
                        'end': [datetime.datetime.fromtimestamp(end_time)],
                        'corr': ["{:.4f}".format(corr_coeff)],
                        'coint': ["{:.4f}".format(coint_coeff)],
                        'stat1': ["{:.6f}".format(stat_coin1)],
                        'stat2': ["{:.6f}".format(stat_coin2)],
                        'stat3': ["{:.6f}".format(stat_coin3)],
                    },
                        index=None)
                    res_df = pd.concat([res_df, new_row], ignore_index=True)
                    res_df.to_csv(r'.\reports\history_check.csv', sep="\t")
                else:
                    print("по монетам данные не получены")


# fill_all_gaps()
# check_parameters_stability(2000, 5, 144, 2)
get_screening_result()
