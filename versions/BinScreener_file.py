import time
import numpy as np
import pandas as pd
import key
import datetime
from os import path
from pathlib import Path
# from typing import Optional
import ccxt
import bin_utils as modul

tf_5m = 5 * 60
tf_5m_str = '5m'
binance = ccxt.binanceusdm({
    'enableRateLimit': True,
    'apiKey': key.binanceAPI,
    'secret': key.binanceSecretAPI,
})

pd.options.mode.chained_assignment = None

result_df = pd.DataFrame(columns=['pair', 'coin1', 'coin2', 'corr', 'coint', 'stat1', 'stat2'], index=None)

exception_list = ['BTSUSDT', 'SCUSDT', 'TLMUSDT', 'BTCSTUSDT', 'FTTUSDT', 'XEMUSDT']


# получить полный список фьючерсов монет с биржи
def get_all_futures():
    binance.load_markets()
    res = binance.markets
    df = pd.DataFrame(res)
    df = df.T
    df = df[df['quote'] == 'USDT']
    df = df.loc[~df['id'].isin(exception_list)]
    # df = df[df['expiry'] == None]
    df = df.sort_values('id')
    return df


# рассчитать коэф корреляции
def get_corr_coeff(coin1, coin2):
    corr_coeff = np.corrcoef(coin1, coin2)
    return corr_coeff


def fill_deep_history_from(start_time):
    # all_coins = get_all_futures()
    # all_futures = all_coins.id.tolist()
    all_futures = ['DARUSDT', 'LRCUSDT', 'LDOUSDT']
    max_time = 1000*tf_5m
    for future in all_futures:
        s_time = start_time
        print(f'Заполняем данные по {future}')
        filename = f"{future}.csv"
        filepath = Path("files", filename)
        # coin_df = pd.DataFrame()
        if path.exists(filepath):
            coin_df = pd.read_csv(filepath, sep="\t")
            if isinstance(coin_df, pd.DataFrame) and len(coin_df) > 0:
                coin_df = modul.prepare_dataframe(df=coin_df, timestamp_field="startTime", asc=False)
                # the oldest event
                base_start = coin_df["time"].values[-1] / 1000
                while base_start > s_time:
                    time_gap = base_start - s_time
                    if time_gap > max_time:
                        time_gap = max_time
                    end_time = s_time + time_gap
                    modul.get_history_price(future, s_time, end_time, tf_5m)
                    s_time = s_time + time_gap
                    time.sleep(1)


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

        coin1_hist = modul.prepare_dataframe(df=coin1_hist, timestamp_field="startTime", asc=True)
        coin2_hist = modul.prepare_dataframe(df=coin2_hist, timestamp_field="startTime", asc=True)
        # получим цены раздвижки
        coin3_hist = modul.make_spread_df(coin1_hist, coin2_hist, last_to_end=True)
        # возьмем только цены закрытия
        close_df1 = coin1_hist.close
        close_df2 = coin2_hist.close
        close_df3 = coin3_hist.close

        # рассчитаем стационарность
        stat_coin1 = modul.stationarity(close_df1)
        stat_coin2 = modul.stationarity(close_df2)
        stat_coin3 = modul.stationarity(close_df3)

        # рассчитаем корреляцию и коинтеграцию
        corr_coeff_df = get_corr_coeff(close_df1, close_df2)
        corr_coeff = corr_coeff_df[0][1]
        coint_coeff = modul.cointegration(close_df1, close_df2)
        # coint_coeff_eg = modul.eg_method(close_df1, close_df2, False)

    add_row = False
    if use_filter:
        # было stat_coin3 < 0.015, немного ужесточил
        if (stat_coin3 < 0.01) and (stat_coin1 > 0.03) and (stat_coin2 > 0.03) \
                and (coint_coeff < 0.05) and (corr_coeff > 0.8):
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


# основная процедура
def get_screening_result(full_control):

    global result_df
    global exception_list

    # получим список всех монет
    all_coins = get_all_futures()
    all_futures = all_coins.id.tolist()
    all_futures2 = all_coins.id.tolist()
    end_time = datetime.datetime.now().timestamp()
    start_time = datetime.datetime.now().timestamp() - 2000 * tf_5m
    for future in all_futures:
        # каждую монету сравним со всеми остальными
        coin1_hist = modul.get_history_price(future, start_time, end_time, tf_5m)
        if len(coin1_hist) > 0:
            print("сравниваем монету " + future)
            for future2 in all_futures2:
                if future2 != future:
                    coin2_hist = modul.get_history_price(future2, start_time, end_time, tf_5m)
                    if len(coin2_hist) > 0:
                        new_row = get_statistics(future, future2, coin1_hist, coin2_hist, True)
                        if new_row is not None:
                            result_df = pd.concat([result_df, new_row], ignore_index=True)
                            print('добавлены данные по паре ' + future + '/' + future2)
                            result_df.to_csv(r'.\screening\1_raw_result.csv', index=False, sep="\t")
                    else:
                        print("по монете "+future2+" данные не получены")
        else:
            print("по монете " + future + " данные не получены")

        all_futures2.remove(future)
    if full_control:
        check_parameters_stability(2000, 5, 72, 3, True)


def fill_history_gaps(coin, tf):
    filename = f"{coin}.csv"
    filepath = Path("files", filename)
    to_add = False
    if path.exists(filepath):
        file_df = pd.read_csv(filepath, sep="\t")
        if isinstance(file_df, pd.DataFrame) and len(file_df) > 0:
            file_df = modul.prepare_dataframe(df=file_df, timestamp_field="startTime", asc=False)
            intervals = modul.get_fetch_intervals(df=file_df, date_column_label="time", timeframe=tf)
            for period in intervals:
                print(f'есть пробелы в {coin}, заполняем')
                s_start = period[1] / 1000
                df = modul.request_history(coin, tf_5m_str, s_start, 1000)
                file_df = pd.concat([file_df, df], ignore_index=True)
                to_add = True

        if isinstance(file_df, pd.DataFrame) and len(file_df) > 0:
            file_df = modul.prepare_dataframe(df=file_df, timestamp_field="startTime", asc=False)
            file_df.to_csv(filepath, columns=["startTime", "time", "open", "high", "low", "close"], index=False,
                           sep="\t")
    return to_add


# проверяет пропуски периодов во всех таблицах данных
def fill_all_gaps(from_list=None, n=1):
    if from_list is None:
        all_coins = get_all_futures()
        from_list = all_coins.id.tolist()
    new_list = []
    for future in from_list:
        print(f'проверка монеты {future}')
        to_add = fill_history_gaps(future, tf_5m)
        if to_add:
            new_list.append(future)
    if (len(new_list) > 0) & (n < 10):
        n = n + 1
        fill_all_gaps(new_list, n)


def check_parameters_stability(window, steps, shift, file, go_next=False):
    """
    :param window: глубина каждого периода проверки (в свечках)
    :param steps: количество шагов проверки
    :param shift: сдвиг назад при каждой новой проверке (в свечках)
    :param file: какой файл проверять
    :param go_next: переходить ли к следующим шагам или нет
    :return:
    """
    res_df = pd.DataFrame()
    if file == 1:
        filepath_check = r'.\reports\bin_to_check.csv'
    elif file == 2:
        filepath_check = r'.\reports\bin_to_check_waiting_list.csv'
    elif file == 3:
        filepath_check = r'.\screening\1_raw_result.csv'
    else:
        filepath_check = r'.\reports\xxx.csv'
    if path.exists(filepath_check):
        check_df = pd.read_csv(filepath_check, sep="\t")
        for index in range(len(check_df)):
            coin1 = check_df.iloc[index]['coin1']
            coin2 = check_df.iloc[index]['coin2']
            print(f'проверяем пару {coin1}/{coin2}')
            for step in range(steps):
                end_time = datetime.datetime.now().timestamp() - step * shift * tf_5m
                start_time = end_time - window * tf_5m
                coin1_hist = modul.get_history_price(coin1, start_time, end_time, tf_5m)
                coin2_hist = modul.get_history_price(coin2, start_time, end_time, tf_5m)

                if len(coin1_hist) > 0 and len(coin2_hist) > 0:
                    new_row = get_statistics(coin1, coin2, coin1_hist, coin2_hist, False)
                    if new_row is not None:
                        res_df = pd.concat([res_df, new_row], ignore_index=True)
                        res_df.to_csv(r'.\screening\2_first_check.csv', index=False, sep="\t")
                else:
                    print("по монетам данные не получены")
    if go_next:
        summarize_history_check(True)


def summarize_history_check(go_next=False):

    filepath_check = r'.\screening\2_first_check.csv'
    if path.exists(filepath_check):
        res_df = pd.read_csv(filepath_check, sep="\t")

        # сгруппируем данные по монетам
        # res_df['pair'] = res_df['coin1'] + '_' + res_df['coin2']
        # res_df.drop(labels=["coin1", "coin2"], axis=1, inplace=True)
        res_df['corr'] = res_df['corr'].astype(float)
        res_df['coint'] = res_df['coint'].astype(float)
        res_df['stat1'] = res_df['stat1'].astype(float)
        res_df['stat2'] = res_df['stat2'].astype(float)
        res_df['stat_pair'] = res_df['stat_pair'].astype(float)
        df = res_df.groupby(['pair']).mean()

        # теперь оставим только удовлетворяющие условиям
        df = df[(df['corr'] >= 0.8) &
                (df['coint'] < 0.05) &
                (df['stat1'] > 0.05) &
                (df['stat2'] > 0.05) &
                (df['stat_pair'] < 0.02)]#было 0.01

        df.to_csv(r'.\screening\3_hard_check.csv', sep="\t")
    else:
        print('файл для анализа не найден')

    if go_next:
        get_historical_strategy_result(False, True)


def get_historical_strategy_result(full_result=True, best_only=True, file=2):
    """

    :param full_result: перебирает по всем вариантам в указанном диапазоне
    :param best_only: для случая если full_result=True - оставляет только лучшие результаты
    :param file: выбор файла
    :return:
    """
    res_df = pd.DataFrame()
    null_df = pd.DataFrame(columns=['pair', 'sma', 'down', 'up', 'result', 'result_per', 'per_no_commis'], index=None)
    # обнулим файл результатов
    null_df.to_csv(r'.\screening\4_optimize_result.csv', index=False, sep="\t")

    if file == 1:
        filepath_check = r'.\reports\bin_to_check.csv'
    elif file == 2:
        filepath_check = r'.\screening\3_hard_check.csv'
    elif file == 3:
        filepath_check = r'.\screening\1_raw_result.csv'
    else:
        filepath_check = r'.\reports\xxx.csv'
    if path.exists(filepath_check):
        res_df = pd.read_csv(filepath_check, sep="\t")

    end = datetime.datetime.now().timestamp()
    # возьмем для теста 2 недели
    start = end - tf_5m*12*24*14

    # use_pair = res_df.columns.isin(['pair']).any()
    for index in range(len(res_df)):
        if 'pair' in res_df.columns:
            pair = res_df.iloc[index]['pair']
            space = pair.find('-')
            coin1 = pair[:space]
            coin2 = pair[space+1:]
        else:
            coin1 = res_df.iloc[index]['coin1']
            coin2 = res_df.iloc[index]['coin2']
            pair = coin1+'_'+coin2

        print(f'рассчитываем {pair}')
        df_coin1 = modul.get_history_price(coin1, start, end, tf_5m)
        df_coin2 = modul.get_history_price(coin2, start, end, tf_5m)
        hist_df = modul.make_spread_df(df_coin1, df_coin2, last_to_end=True)

        if full_result:
            ########################################
            # блок условий
            up_from, up_to = 1.5, 4.0
            down_from, down_to = -1.5, -4.0
            step, step_sma = 0.5, 50
            sma_from, sma_to = 50, 300

            temp_up = up_from
            while temp_up <= up_to:
                temp_down = down_to
                while temp_down <= down_from:
                    temp_sma = sma_from
                    while temp_sma <= sma_to:
                        # print(f'  sma={temp_sma}, Up={temp_up}, Down={temp_down}')
                        modul.calculate_historical_profit(hist_df, pair, temp_sma, temp_up, temp_down)
                        temp_sma = temp_sma + step_sma
                    temp_down = temp_down + step
                temp_up = temp_up + step
        else:
            modul.calculate_historical_profit(hist_df, pair, sma=100, up=2.0, down=-2.0)

    if best_only:
        total_df = pd.DataFrame()
        final_df = pd.DataFrame()
        filepath_check = r'.\screening\4_optimize_result.csv'
        if path.exists(filepath_check):
            total_df = pd.read_csv(filepath_check, sep="\t")
        for index in range(len(res_df)):
            pair = res_df.iloc[index]['pair']
            pair_df = total_df[total_df.pair == pair]
            max_res = pair_df['per_no_commis'].max()
            new_row = pair_df[pair_df.per_no_commis == max_res]
            final_df = pd.concat([final_df, new_row], ignore_index=True)
        final_df.to_csv(r'.\screening\4_optimize_result.csv', index=False, sep="\t")

    print('расчет исторических результатов торгов закончен')


# основная процедура, при указании True запустятся последующие процедуры
# get_screening_result(True)


# блок отдельных процедур, для тестирования
# check_parameters_stability(2000, 5, 144, 1, False)
# summarize_history_check()
# get_historical_strategy_result(False, True, 2) # 3_hard_check
# get_historical_strategy_result(True, True, 4) # полный расчет, xxx


# блок первичного заполнения исторических данных
start_t = datetime.datetime(2022, 11, 10, 0, 0).timestamp()
fill_deep_history_from(start_t)

# запускать иногда, раз в неделю-две
fill_all_gaps()
