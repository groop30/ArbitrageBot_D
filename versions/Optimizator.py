import datetime
import pandas as pd
import numpy as np
import talib
from old_FTX.FTXclient import FtxClient
# import scipy.stats as stats
from os import path
from pathlib import Path

base_url = 'https://ftx.com/api/'

ftx_client = FtxClient(
    api_key='',
    api_secret='')
pd.options.mode.chained_assignment = None


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
    if concat_needed:
        # запрашиваем из базы только, если объединение нужно
        res = ftx_client.get_historical_prices(asset, tf, s_start, end)
        df = pd.DataFrame(res)
        df = pd.concat([history_df, df], ignore_index=True)
    else:
        df = history_df

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


def calculate_history(hist_df, pair, sma, up, down, tf):

    # going_to = ''
    # op_price = 0.0
    # cl_price = 0.0
    # l_result = 0.0
    # l_result_per = 0.0
    # op_time = ''
    # cl_time = ''
    # position_open = False
    # save_result = False
    # up_str = str(up)
    # up_str = up_str.replace(".", "_")
    # down_str = str(down)
    # down_str = down_str.replace(".", "_")
    # filename = f'optimize_{pair}_sma{sma}_up{up_str}_down{down_str}.csv'
    # filepath = Path("optimization", filename)
    # result_df = pd.DataFrame()
    # total = 0.0
    # total_per = 0.0
    # per_no_commis = 0.0

    # for index in range(len(hist_df)):
    #     # сначала рассчитаем параметры zscore
    #     end_time = hist_df.iloc[index]['time_x']
    #     start_time = end_time - sma * tf*1000
    #     df = hist_df
    #     df = df[(df.time_x <= end_time) & (df.time_x >= start_time)]
    #     if len(df) < sma:
    #         continue
    #     df['bb_up'], df['sma'], df['bb_down'] = talib.BBANDS(df.close, sma, 2, 2, 0)
    #     df['zscore'] = stats.zscore(df.close)
    #
    #     # выделим нужное значение
    #     last_row = df.tail(1)
    #     # l_sma = last_row.iloc[0]['sma']
    #     # l_zscore = last_row.iloc[0]['zscore']
    #     # l_price = hist_df.iloc[index]['close']
    #     # l_time = hist_df.iloc[index]['time_x']
    #     l_time, l_price, l_sma, l_zscore = (
    #         last_row[c].to_numpy()[0]
    #         for c in last_row
    #         if c in ["time_x", "close", "sma", "zscore"]
    #     )
    #     # потом проверяем выход за нужные параметры
    #     if position_open:
    #         if (l_sma > l_price) & (going_to == 'DOWN'):
    #             cl_time = datetime.datetime.fromtimestamp(l_time/1000)
    #             cl_price = l_price
    #             l_result = op_price - cl_price
    #             l_result_per = (l_result/cl_price*100)
    #             total = total + l_result
    #             total_per = total_per + l_result_per
    #             per_no_commis = total_per - 0.252
    #             position_open = False
    #             save_result = True
    #         elif (l_sma < l_price) & (going_to == 'UP'):
    #             cl_time = datetime.datetime.fromtimestamp(l_time/1000)
    #             cl_price = l_price
    #             l_result = cl_price - op_price
    #             l_result_per = (l_result / cl_price * 100)
    #             total = total + l_result
    #             total_per = total_per + l_result_per
    #             per_no_commis = total_per - 0.252
    #             position_open = False
    #             save_result = True
    #     else:
    #         if l_zscore > up:
    #             # открываем шорт
    #             op_time = datetime.datetime.fromtimestamp(l_time/1000)
    #             op_price = l_price
    #             position_open = True
    #             going_to = 'DOWN'
    #
    #         elif l_zscore < down:
    #             # открываем long
    #             op_time = datetime.datetime.fromtimestamp(l_time/1000)
    #             op_price = l_price
    #             position_open = True
    #             going_to = 'UP'
    #
    #     # сохраняем результат
    #     if save_result:
    #         new_row = pd.DataFrame({
    #             'pair': [pair],
    #             'op_time': [op_time],
    #             'op_price': ["{:.6f}".format(op_price)],
    #             'going_to': [going_to],
    #             'cl_time': [cl_time],
    #             'cl_price': ["{:.6f}".format(cl_price)],
    #             'result': ["{:.6f}".format(l_result)],
    #             'result_per': ["{:.3f}".format(l_result_per)]
    #         },
    #             index=None)
    #         result_df = pd.concat([result_df, new_row], ignore_index=True)
    #         result_df.to_csv(filepath, index=False, sep="\t")
    #         save_result = False

    # TODO - переписать расчет без перебора строк
    hist_df = zscore_calculating(hist_df, sma)
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
    # hist_df.to_csv(r'.\optimization\full_data.csv', index=False, sep="\t")
    # остальные строки удаляем, как не нужные, сдвигаем дф еще раз
    hist_df = hist_df[
        (hist_df.going_to == 'zero') | (hist_df.going_to == 'UP') | (hist_df.going_to == 'DOWN')]
    hist_df['cross_shift'] = hist_df.shift(periods=1)['going_to']
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
    hist_df['result_per_no'] = hist_df['result_per'] - 0.252
    total = hist_df['result'].sum()
    total_per = hist_df['result_per'].sum()
    per_no_commis = hist_df['result_per_no'].sum()

    # hist_df.to_csv(r'.\optimization\new_optimize.csv', index=False, sep="\t")

    total_row = pd.DataFrame({
        'pair': [pair],
        'sma': [sma],
        'up': [up],
        'down': [down],
        'result': ["{:.6f}".format(total)],
        'result_per': ["{:.3f}".format(total_per)],
        'per_no_commis': ["{:.3f}".format(per_no_commis)]},
        index=None)

    filepath_total = r'.\optimization\total_optimize.csv'
    if path.exists(filepath_total):
        # добавим к имеющимся парам для отслеживания новые
        total_df = pd.read_csv(filepath_total, sep="\t")
        total_df = pd.concat([total_df, total_row], ignore_index=True)
        total_df.to_csv(filepath_total, index=False, sep="\t")
    else:
        total_row.to_csv(filepath_total, index=False, sep="\t")


# основная процедура
def main():
    opt_list = pd.DataFrame(columns=["coin1", "coin2"])
    filepath_close = r'.\reports\to_optimize.csv'
    if path.exists(filepath_close):
        opt_list = pd.read_csv(filepath_close, sep="\t")
    # обнулим файл с итогами
    # filepath_total = r'.\optimization\total_optimize.csv'
    # total_row = pd.DataFrame()
    # total_row.to_csv(filepath_total, index=False, sep="\t")
    ########################################
    # блок условий
    up_from = 2.0
    up_to = 4.0
    down_from = -2.0
    down_to = -4.0
    step = 0.5
    step_sma = 50
    sma_from = 50
    sma_to = 250
    start = datetime.datetime(2022, 10, 5, 0, 0, 0).timestamp()
    end = datetime.datetime(2022, 11, 4, 0, 0, 0).timestamp()
    # forward = True
    # size = 1000.0
    tf = 5*60

    for index in range(len(opt_list)):
        coin1 = opt_list.iloc[index]['coin1']
        coin2 = opt_list.iloc[index]['coin2']
        pair = f'{coin1}_{coin2}'
        print(f'рассчитываем {pair}')
        hist_df = make_spead_df(coin1, coin2, tf, start, end)
        temp_up = up_from
        while temp_up <= up_to:
            temp_down = down_to
            while temp_down <= down_from:
                temp_sma = sma_from
                while temp_sma <= sma_to:
                    print(f'  sma={temp_sma}, Up={temp_up}, Down={temp_down}')
                    calculate_history(hist_df, pair, temp_sma, temp_up, temp_down, tf)
                    temp_sma = temp_sma + step_sma
                temp_down = temp_down + step
            temp_up = temp_up + step


main()
