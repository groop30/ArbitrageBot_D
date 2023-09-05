import time
import numpy as np
import pandas as pd
import key
import datetime
from os import path
from pathlib import Path
# import schedule
import ccxt
import bin_utils as modul
import talib

tf_5m = 5 * 60
tf_5m_str = '5m'
binance = ccxt.binanceusdm({
    'enableRateLimit': True,
    'apiKey': key.binanceAPI,
    'secret': key.binanceSecretAPI,
})
alerts = []
pd.options.mode.chained_assignment = None

result_df = pd.DataFrame(columns=['pair', 'coin1', 'coin2', 'corr', 'coint', 'stat1', 'stat2'], index=None)
fill_gap_list = ['indx_11']
connection = modul.connect_to_sqlalchemy_binance()

# исключим монеты с маленьким плечом (меньше 20)
exception_list = ['AMBUSDT', 'COCOSUSDT', 'CVCUSDT', 'FTTUSDT', 'HNTUSDT', 'SRMUSDT', 'STMXUSDT',
                  'KSOS-PERP', 'SOS-PERP', 'MVDA10-PERP', 'MVDA25-PERP', 'BTT-PERP', 'KBTT-PERP', 'PAXG-PERP',
                  'MVDA25-PERP', 'PAXG-PERP', 'RON-PERP', 'UNISWAP-PERP', 'XAUT-PERP', 'YFII-PERP']

# ###########################################################################
# Блок заполнения исторических данных
#
# ###########################################################################
def fill_deep_history_from(start_time):
    all_coins = modul.get_all_futures()
    all_futures = all_coins.id.tolist()
    # all_futures = ['DUSKUSDT', 'LINAUSDT']
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


def fill_deep_sql_history_from(start_time):
    all_coins = modul.get_all_futures()
    all_futures = all_coins.id.tolist()
    # all_futures = ['DUSKUSDT', 'LINAUSDT']
    max_time = 1000*tf_5m

    for future in all_futures:
        s_time = start_time
        print(f'Заполняем данные по {future}')
        coin_table = modul.create_olhc_table(future, connection)
        query = coin_table.select()
        with connection.connect() as conn:
            coin_df = pd.read_sql(sql=query, con=conn)

        # coin_df = pd.DataFrame()
        if len(coin_df) > 0:
            coin_df = modul.prepare_dataframe(df=coin_df, timestamp_field="startTime", asc=False)
            # the oldest event
            base_start = coin_df["time"].values[-1] / 1000
            while base_start > s_time:
                time_gap = base_start - s_time
                if time_gap > max_time:
                    time_gap = max_time
                end_time = s_time + time_gap
                modul.get_sql_history_price(future, connection, s_time, end_time)
                s_time = s_time + time_gap
                time.sleep(1)


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


def fill_sql_history_gaps(coin, tf):
    coin_table = modul.create_olhc_table(coin, connection)
    query = coin_table.select()
    conn = connection.connect()
    file_df = pd.read_sql(sql=query, con=conn)
    to_add = False
    if isinstance(file_df, pd.DataFrame) and len(file_df) > 0:
        file_df = modul.prepare_dataframe(df=file_df, timestamp_field="startTime", asc=False)
        intervals = modul.get_fetch_intervals(df=file_df, date_column_label="time", timeframe=tf)
        if len(intervals) > 0:
            print(f'есть пробелы в {coin}, заполняем')
            s_start = intervals[0][1] / 1000
            modul.add_candels_to_database(coin, file_df, s_start, 1000, conn)
            to_add = True

    return to_add


# проверяет пропуски периодов во всех таблицах данных
def fill_all_gaps(from_list=None, n=1):
    if from_list is None:
        all_coins = modul.get_all_futures()
        from_list = all_coins.id.tolist()
    new_list = []
    for future in from_list:
        print(f'проверка монеты {future}')
        to_add = fill_sql_history_gaps(future, tf_5m)
        if to_add:
            new_list.append(future)
    if (len(new_list) > 0) & (n < 10):
        n = n + 1
        fill_all_gaps(new_list, n)


# ###########################################################################
# Блок процедур ежедневного сканирования и поиска пар для торговли
#
# ###########################################################################
# STEP 1
def get_screening_result(full_control, lookback, end_time=None):

    result_df = pd.DataFrame(columns=['pair', 'coin1', 'coin2', 'corr', 'coint', 'stat1', 'stat2'], index=None)

    # получим список всех монет
    all_coins = modul.get_all_futures()
    all_coins = all_coins.loc[~all_coins['id'].isin(exception_list)]
    all_futures = all_coins.id.tolist()
    all_futures2 = all_coins.id.tolist()
    list_for_remove2 = []
    if end_time is None:
        end_time = datetime.datetime.now().timestamp()
        start_time = datetime.datetime.now().timestamp() - lookback * tf_5m
        filepath = r'.\screening\1_raw_result.csv'
    else:
        start_time = end_time - lookback * tf_5m
        filepath = r'.\testing\1_history_raw_result.csv'
        # filepath = r'.\screening\1_raw_result.csv'

    for future in all_futures:
        # каждую монету сравним со всеми остальными
        # coin1_hist = modul.get_history_price(future, start_time, end_time, tf_5m)
        coin1_hist = modul.get_sql_history_price(future, connection, start_time, end_time)
        if len(coin1_hist) > 0:
            # coin1_hist = modul.convert_to_tf(coin1_hist, 900)
            stat_coin1 = modul.stationarity(coin1_hist['close'])
            if stat_coin1 < 0.05:
                all_futures2.remove(future)
                continue
            print("Шаг 1 - сравниваем монету " + future)
            all_futures2 = [x for x in all_futures2 if x not in list_for_remove2]
            list_for_remove2 = []
            for future2 in all_futures2:
                if future2 != future:
                    time3 = time.time()
                    # coin2_hist = modul.get_history_price(future2, start_time, end_time, tf_5m)
                    coin2_hist = modul.get_sql_history_price(future2, connection, start_time, end_time)
                    if len(coin2_hist) > 0:
                        # coin2_hist = modul.convert_to_tf(coin2_hist, 900)
                        # print('проверяем пару ' + future + '/' + future2)
                        new_row = modul.get_statistics(future, future2, coin1_hist, coin2_hist, True)
                        # e_distance = modul.euclidean_distance(coin1_hist, coin2_hist)
                        if new_row is not None:
                            result_df = pd.concat([result_df, new_row], ignore_index=True)
                            print('добавлены данные по паре ' + future + '/' + future2)
                            # result_df.to_csv(filepath, index=False, sep="\t)
                    else:
                        print("по монете "+future2+" данные не получены")
                        list_for_remove2.append(future2)
        else:
            print("по монете " + future + " данные не получены")

        if future in all_futures2:
            all_futures2.remove(future)

    result_df.to_csv(filepath, index=False, sep="\t")

    if full_control:
        check_parameters_stability(2000, 5, 72, 3, True)


# STEP 1.2 (пока не используется)
def get_screening_to_index(lookback):
    global result_df
    # получим список всех монет
    indx_name = 'indx_11'
    all_coins = modul.get_all_futures()
    all_futures = all_coins.id.tolist()
    end_time = datetime.datetime.now().timestamp()
    start_time = datetime.datetime.now().timestamp() - lookback * tf_5m
    indx_hist = modul.get_index_history(indx_name, connection, start_time, end_time)
    for future in all_futures:
        print("сравниваем монету " + future)
        coin2_hist = modul.get_sql_history_price(future, connection, start_time, end_time)
        if len(coin2_hist) > 0:
            new_row = modul.get_statistics(indx_name, future, indx_hist, coin2_hist, True)
            if new_row is not None:
                result_df = pd.concat([result_df, new_row], ignore_index=True)
                print('добавлены данные по паре ' + indx_name + '/' + future)

        else:
            print("по монете " + future + " данные не получены")

    result_df.to_csv(r'.\screening\1_raw_result.csv', index=False, sep="\t")


# STEP 2 and 5
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
        filepath_check = r'.\screening\3_hard_check.csv'
    elif file == 3:
        filepath_check = r'.\screening\1_raw_result.csv'
    elif file == 4:
        filepath_check = r'.\testing\1_history_raw_result.csv'
    else:
        filepath_check = r'.\reports\xxx.csv'
    if path.exists(filepath_check):
        check_df = pd.read_csv(filepath_check, sep="\t")
        for index in range(len(check_df)):
            coin1 = check_df.iloc[index]['coin1']
            coin2 = check_df.iloc[index]['coin2']
            print(f'Шаг 2 - проверяем пару {coin1}/{coin2}')
            for step in range(steps):
                end_time = datetime.datetime.now().timestamp() - step * shift * tf_5m
                start_time = end_time - window * tf_5m
                coin1_hist = modul.get_sql_history_price(coin1, connection, start_time, end_time)
                # coin1_hist = modul.convert_to_tf(coin1_hist, 900)
                coin2_hist = modul.get_sql_history_price(coin2, connection, start_time, end_time)
                # coin2_hist = modul.convert_to_tf(coin2_hist, 900)
                if len(coin1_hist) > 0 and len(coin2_hist) > 0:
                    new_row = modul.get_statistics(coin1, coin2, coin1_hist, coin2_hist, False)
                    if new_row is not None:
                        res_df = pd.concat([res_df, new_row], ignore_index=True)
                else:
                    print("по монетам данные не получены")
        if file == 1:
            res_df.to_csv(r'.\screening\5_actual_pairs_stability_check.csv', index=False, sep="\t")
        elif file == 4:
            res_df.to_csv(r'.\testing\2_history_first_check.csv', index=False, sep="\t")
        else:
            res_df.to_csv(r'.\screening\2_first_check.csv', index=False, sep="\t")

    if go_next:
        if file == 1:
            summarize_actual_pairs_stability(True)
        else:
            summarize_history_check(True)


# STEP 3
def summarize_history_check(go_next=False, file=1):
    if file == 2:
        filepath_check = r'.\testing\2_history_first_check.csv'
    else:
        filepath_check = r'.\screening\2_first_check.csv'

    if path.exists(filepath_check):
        res_df = pd.read_csv(filepath_check, sep="\t")

        # сгруппируем данные по монетам
        res_df['corr'] = res_df['corr'].astype(float)
        res_df['coint'] = res_df['coint'].astype(float)
        res_df['stat1'] = res_df['stat1'].astype(float)
        res_df['stat2'] = res_df['stat2'].astype(float)
        res_df['stat_pair'] = res_df['stat_pair'].astype(float)
        df = res_df.groupby(['pair']).mean(numeric_only=True)

        # теперь оставим только удовлетворяющие условиям
        df = df[(df['corr'] >= 0.8) &
                (df['coint'] < 0.05) &
                (df['stat1'] > 0.05) &
                (df['stat2'] > 0.05) &
                (df['stat_pair'] < 0.02)]
        # df = df[(df['coint'] > 0.05) &
        #         (df['coint'] < 0.2) &
        #         (df['stat1'] > 0.05) &
        #         (df['stat2'] > 0.05) &
        #         (df['stat_pair'] < 0.03)]
        if file == 2:
            df.to_csv(r'.\testing\3_history_hard_check.csv', sep="\t")
        else:
            df.to_csv(r'.\screening\3_hard_check.csv', sep="\t")
    else:
        print('файл для анализа не найден')

    if go_next:
        get_historical_strategy_result(False, True)


# STEP 6
def summarize_actual_pairs_stability(go_next=False):
    filepath_check = r'.\screening\5_actual_pairs_stability_check.csv'
    filepath_act_pairs = r'.\reports\bin_to_check.csv'
    list_df = pd.read_csv(filepath_act_pairs, sep="\t")
    if path.exists(filepath_check):
        res_df = pd.read_csv(filepath_check, sep="\t")

        # сгруппируем данные по монетам
        res_df['corr'] = res_df['corr'].astype(float)
        res_df['coint'] = res_df['coint'].astype(float)
        res_df['stat1'] = res_df['stat1'].astype(float)
        res_df['stat2'] = res_df['stat2'].astype(float)
        res_df['stat_pair'] = res_df['stat_pair'].astype(float)
        df = res_df.groupby(['pair'], as_index=False).mean()

        list_df['pair'] = list_df['coin1'] + '-' + list_df['coin2']

        df.sort_values(by='pair', ascending=True, inplace=True, ignore_index=True)
        list_df.sort_values(by='pair', ascending=True, inplace=True, ignore_index=True)

        list_df['coint'] = np.where(list_df['pair'] == df['pair'], df['coint'], 'NO')
        list_df['stat'] = np.where(list_df['pair'] == df['pair'], df['stat_pair'], 'NO')

        list_df.to_csv(r'.\screening\7_final_result.csv', index=False, sep="\t")
    else:
        print('файл для анализа не найден')

    if go_next:
        get_historical_strategy_result(full_result=True, best_only=True, file=1, days=7)


# STEP 4 and 7 (Переписать!! Расчет идет по старой стратегии)
def get_historical_strategy_result(full_result=True, best_only=True, file=2, days=14, cut_df=500):
    """
    Процедура получает результат стратегии с перебором по разным параметрам.
    Дальше выбирает лучшие результаты (если best_only=True), и обрезает остальные (cut_df=)
    :param full_result: перебирает по всем вариантам в указанном диапазоне
    :param best_only: для случая если full_result=True - оставляет только лучшие результаты
    :param file: выбор файла
    :param days: количество дней для тестирования
    :param cut_df: на сколько обрезать выборку (для файла №2)
    :return:
    """
    res_df = pd.DataFrame()

    if file == 1:
        filepath_check = r'.\reports\bin_to_check.csv'
    elif file == 2:
        filepath_check = r'.\screening\3_hard_check.csv'
    elif file == 3:
        filepath_check = r'.\screening\1_raw_result.csv'
    elif file == 4:
        filepath_check = r'.\screening\4_optimize_result.csv'
    else:
        filepath_check = r'.\reports\xxx.csv'

    if path.exists(filepath_check):
        res_df = pd.read_csv(filepath_check, sep="\t")
        res_df = res_df.head(cut_df)

    # обнулим файл результатов
    if file == 1:
        filepath_to = r'.\screening\6_actual_pairs_optimize_result.csv'
    else:
        filepath_to = r'.\screening\4_optimize_result.csv'
    total_df = pd.DataFrame(columns=['pair', 'sma', 'down', 'up', 'result', 'result_per', 'per_no_commis'], index=None)
    total_df.to_csv(filepath_to, index=False, sep="\t")

    end = datetime.datetime.now().timestamp()
    # возьмем для теста 2 недели
    start = end - tf_5m*12*24*days

    for index in range(len(res_df)):
        if 'pair' in res_df.columns:
            pair = res_df.iloc[index]['pair']
            space = pair.find('-')
            coin1 = pair[:space]
            coin2 = pair[space+1:]
        else:
            coin1 = res_df.iloc[index]['coin1']
            coin2 = res_df.iloc[index]['coin2']
            pair = coin1+'-'+coin2

        print(f'рассчитываем {pair}')
        df_coin1 = modul.get_sql_history_price(coin1, connection, start, end)
        df_coin2 = modul.get_sql_history_price(coin2, connection, start, end)
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
                        total_row = modul.calculate_historical_profit(hist_df, pair, 'zscore',
                                                                      temp_sma, temp_up, temp_down)
                        total_df = pd.concat([total_df, total_row], ignore_index=True)
                        temp_sma = temp_sma + step_sma
                    temp_down = temp_down + step
                temp_up = temp_up + step

        else:
            total_row = modul.calculate_historical_profit(hist_df, pair, 'zscore', sma=100, up=2.0, down=-2.0)
            total_df = pd.concat([total_df, total_row], ignore_index=True)

    if best_only:
        # total_df = pd.DataFrame()
        final_df = pd.DataFrame()
        # total_df = pd.read_csv(filepath_to, sep="\t")
        total_df['per_no_commis'] = total_df['per_no_commis'].astype(float)
        total_df['result_per'] = total_df['result_per'].astype(float)
        for index in range(len(res_df)):
            if 'pair' in res_df.columns:
                pair = res_df.iloc[index]['pair']
            else:
                coin1 = res_df.iloc[index]['coin1']
                coin2 = res_df.iloc[index]['coin2']
                pair = coin1 + '-' + coin2
            pair_df = total_df[total_df.pair == pair]
            max_res = pair_df['per_no_commis'].max()
            new_row = pair_df[pair_df.per_no_commis == max_res]
            # если после отсева все равно осталось несколько строк - берем первую
            final_df = pd.concat([final_df, new_row.head(1)], ignore_index=True)
        final_df.sort_values(
          by='per_no_commis',
          ascending=False,
          inplace=True,
          ignore_index=True,
        )
        final_df.to_csv(filepath_to, index=False, sep="\t")
    else:
        total_df.to_csv(filepath_to, index=False, sep="\t")

    if file == 2:
        # идет полная проверка, сделаем отбор первых 50 лучших результатов,
        # и проведем полную оптимизацию по последним 5ти дням
        get_historical_strategy_result(full_result=True, best_only=True, file=4, days=7, cut_df=50)
    elif file == 1:
        calculate_trades_result()

    print('расчет исторических результатов торгов закончен')


# STEP 8
def calculate_trades_result():
    filepath_act_presult = r'.\screening\7_final_result.csv'
    full_df = pd.read_csv(filepath_act_presult, sep="\t")
    filepath_opti_presult = r'.\screening\6_actual_pairs_optimize_result.csv'
    opti_df = pd.read_csv(filepath_opti_presult, sep="\t")
    filepath_log = r'.\reports\bin_to_log.csv'
    log_df = pd.read_csv(filepath_log, sep="\t")

    full_df.sort_values(by='pair', ascending=True, inplace=True, ignore_index=True)
    opti_df.sort_values(by='pair', ascending=True, inplace=True, ignore_index=True)
    full_df['new_sma'] = np.where(full_df['pair'] == opti_df['pair'], opti_df['sma'], 'NO')
    full_df['new_down'] = np.where(full_df['pair'] == opti_df['pair'], opti_df['down'], 'NO')
    full_df['new_up'] = np.where(full_df['pair'] == opti_df['pair'], opti_df['up'], 'NO')
    full_df['plan_profit'] = np.where(full_df['pair'] == opti_df['pair'], opti_df['per_no_commis'], 'NO')

    # Выберем данные из лога за последние 5 дней
    day_now = datetime.datetime.today()
    day_now = day_now.combine(day_now.date(), day_now.min.time())
    day_from = day_now - datetime.timedelta(5)

    log_df['pair'] = log_df['coin1'] + '-' + log_df['coin2']
    log_df['op_time'] = pd.to_datetime(log_df['op_time'], format="%Y-%m-%d %H:%M:%S")
    filter_df = log_df[log_df['op_time'] > day_from]

    group_df = filter_df.groupby(['pair'], as_index=False).sum()
    group_df['day_profit'] = round(group_df['per_no_commis'] / 5, 3)
    group_df.sort_values(by='pair', ascending=True, inplace=True, ignore_index=True)
    # на этом этапе делаем перебор строк.
    full_df['fact_profit'] = 0.0
    for index in range(len(group_df)):
        pair = group_df.iloc[index]['pair']
        day_profit = group_df.iloc[index]['day_profit']
        pair_df = full_df.loc[full_df['pair'] == pair]
        if len(pair_df) > 0:
            index_row = pair_df.index[0]
            full_df.at[index_row, 'fact_profit'] = day_profit

    full_df.to_csv(filepath_act_presult, index=False, sep="\t")
    print('Все рассчеты окончены!')


# Экспериментальная процедура, дистанционный и прочие методы вместе.
def get_triangle_selected_list(lookback):
    # global result_df
    # global exception_list
    res_df = pd.DataFrame()
    # получим список всех монет
    all_coins = modul.get_all_futures()
    all_futures = all_coins.id.tolist()
    all_futures2 = all_coins.id.tolist()
    end_time = datetime.datetime.now().timestamp()
    start_time = datetime.datetime.now().timestamp() - lookback * tf_5m
    for future in all_futures:
        # каждую монету сравним со всеми остальными
        coin1_hist = modul.get_sql_history_price(future, connection, start_time, end_time)
        if len(coin1_hist) < (lookback - 10):
            # Если нет данных за весь период, то проверку не делаем
            all_futures2.remove(future)
            continue
        stat_coin1 = modul.stationarity(coin1_hist['close'])
        if stat_coin1 < 0.05:
            all_futures2.remove(future)
            continue
        if len(coin1_hist) > 0:
            print("сравниваем монету " + future)
            for future2 in all_futures2:
                if future2 != future:
                    coin2_hist = modul.get_sql_history_price(future2, connection, start_time, end_time)
                    if len(coin2_hist) < (lookback - 10):
                        # Если нет данных за весь период, то проверку не делаем
                        continue
                    stat_coin2 = modul.stationarity(coin2_hist['close'])
                    if stat_coin2 < 0.05:
                        continue
                    if len(coin2_hist) > 0:
                        # e_distance = modul.euclidean_distance(coin1_hist, coin2_hist)
                        new_row = modul.get_statistics(future, future2, coin1_hist, coin2_hist, False)
                        if float(new_row.iloc[0]['coint']) < 0.1:
                            df = modul.make_spread_df(coin1_hist, coin2_hist, last_to_end=True, tf=tf_5m)
                            time_round, time_to_end = modul.check_for_touch_bb(df, 1000, 1)

                            dev_df = modul.get_max_deviation_from_sma(df, 1000)
                            try:
                                max_dev = dev_df['max_deviation'].max()
                                mean_dev = dev_df['max_deviation'].mean()
                                abnorm_tri = dev_df['abnormal_count'].sum()
                                low_dev = dev_df['low_count'].sum()
                                hl_dev = dev_df['hl_count'].sum()
                                all_counts = len(dev_df)
                                norm_df = dev_df[dev_df['norm_count'] == 1]
                                level_80 = dev_df['max_deviation'].quantile(0.8)
                                norm_lvl_80 = norm_df['max_deviation'].quantile(0.8)
                                # отбор для класики
                                # if (4.0 < max_dev < 9.0) and (hl_dev < 3) and (all_counts > len(df)/100):
                                if 4.0 < max_dev < 20.0:  # отбор для выносов
                                    # new_row['dist'] = round(e_distance, 4)
                                    new_row['norm_triangle'] = all_counts
                                    new_row['bad_triangle'] = abnorm_tri
                                    new_row['low_dev'] = low_dev
                                    new_row['out_time'] = hl_dev
                                    new_row['max_dev'] = max_dev
                                    new_row['mean_dev'] = mean_dev
                                    new_row['level_80'] = level_80
                                    new_row['norm_lvl_80'] = norm_lvl_80
                                    # temp_df = pd.concat([new_row, new_row2], ignore_index=True)
                                    res_df = pd.concat([res_df, new_row], ignore_index=True)
                                    print('добавлены данные по паре ' + future + '/' + future2)
                            except:
                                pass
                    else:
                        print("по монете " + future2 + " данные не получены")
        else:
            print("по монете " + future + " данные не получены")

        all_futures2.remove(future)

    res_df.to_csv(r'.\screening\1_dist_result.csv', index=False, sep="\t")


def get_bb_touch_list(lookback):

    res_df = pd.DataFrame()
    # получим список всех монет
    all_coins = modul.get_all_futures()
    all_futures = all_coins.id.tolist()
    all_futures2 = all_coins.id.tolist()
    end_time = datetime.datetime.now().timestamp()
    start_time = datetime.datetime.now().timestamp() - lookback * 2 * tf_5m

    halftime = lookback/2
    for future in all_futures:
        # каждую монету сравним со всеми остальными
        coin1_hist = modul.get_sql_history_price(future, connection, start_time, end_time)
        if len(coin1_hist) < (lookback - 10):
            # Если нет данных за весь период, то проверку не делаем
            all_futures2.remove(future)
            continue
        stat_coin1 = modul.stationarity(coin1_hist['close'])
        if stat_coin1 < 0.05:
            all_futures2.remove(future)
            continue
        if len(coin1_hist) > 0:
            print("сравниваем монету " + future)
            for future2 in all_futures2:
                if future2 != future:
                    coin2_hist = modul.get_sql_history_price(future2, connection, start_time, end_time)
                    if len(coin2_hist) < (lookback - 10):
                        # Если нет данных за весь период, то проверку не делаем
                        continue
                    stat_coin2 = modul.stationarity(coin2_hist['close'])
                    if stat_coin2 < 0.05:
                        continue
                    if len(coin2_hist) > 0:
                        df = modul.make_spread_df(coin1_hist, coin2_hist, last_to_end=True, tf=tf_5m)
                        time_round, time_to_end = modul.check_for_touch_bb(df, lookback, 1)
                        if time_round < halftime and time_to_end < halftime:
                            new_row = pd.DataFrame({
                                'pair': [future + '/' + future2],
                                'round': [time_round],
                                'to_end': [time_to_end]
                            },
                                index=None)

                            res_df = pd.concat([res_df, new_row], ignore_index=True)
                    else:
                        print("по монете " + future2 + " данные не получены")
        else:
            print("по монете " + future + " данные не получены")

        all_futures2.remove(future)

    res_df.to_csv(r'.\screening\1_touch_result.csv', index=False, sep="\t")


def check_for_touch_bb(lookback, sigma, file):

    if file == 1:
        filepath_check = r'.\screening\1_raw_result.csv'
    elif file == 2:
        filepath_check = r'.\screening\3_hard_check.csv'
    elif file == 3:
        filepath_check = r'.\testing\1_history_raw_result.csv'
    else:
        filepath_check = r'.\reports\xxx.csv'

    end_time = datetime.datetime.now().timestamp()
    start_time = datetime.datetime.now().timestamp() - lookback * 2 * tf_5m
    halftime = lookback / 2
    res_df = pd.DataFrame()

    if path.exists(filepath_check):
        check_df = pd.read_csv(filepath_check, sep="\t")
        for index in range(len(check_df)):
            if "coin1" in check_df.columns:
                coin1 = check_df.iloc[index]['coin1']
                coin2 = check_df.iloc[index]['coin2']
            else:
                pair = check_df.iloc[index]['pair']
                coin1, coin2 = modul.pair_to_coins(pair)

            print(f'Шаг 3 - проверяем пару {coin1}/{coin2}')
            coin1_df = modul.get_sql_history_price(coin1, connection, start_time, end_time)
            coin2_df = modul.get_sql_history_price(coin2, connection, start_time, end_time)
            spread_df = modul.make_spread_df(coin1_df, coin2_df, last_to_end=True, tf=tf_5m)
            time_round, time_to_end, time_to_opposite = modul.check_for_touch_bb(spread_df, lookback, sigma)
            if time_to_opposite < halftime:
                new_row = pd.DataFrame({
                    'pair': [coin1 + '-' + coin2],
                    'coin1': [coin1],
                    'coin2': [coin2],
                    'coint':[check_df.iloc[index]['coint']],
                    'stat_pair': [check_df.iloc[index]['stat_pair']],
                    'round': [time_round],
                    'to_end': [time_to_end]
                },
                    index=None)

                res_df = pd.concat([res_df, new_row], ignore_index=True)

    if file == 2:
        res_df.to_csv(r'.\screening\bb_touch_result.csv', index=False, sep="\t")


# ######################################################################################
# Блок процедур для постоянного сканирования и перерасчета по расписанию
#
# ###########################################################################
def calculate_checkpairs_position():
    """
    Обновляет рассчетные колонки в таблице пар для регулярной проверки
    :return:
    """
    check_df = modul.get_selected_pairs(connection)
    check_df.sort_values(by='pair', ascending=True, inplace=True, ignore_index=True)
    end_time = datetime.datetime.now().timestamp()
    start_time = datetime.datetime.now().timestamp() - 240 * tf_5m - tf_5m * 50
    for index in range(len(check_df)):
        # получим данные строки
        pair, coin1, coin2, up, down = (
            check_df[c].to_numpy()[index]
            for c in check_df
            if c in ["pair", "coin1", "coin2", "up", "down"]
        )
        # запросим исторические данные по монетам
        df_coin1 = modul.get_sql_history_price(coin1, connection, start_time, end_time)
        df_coin2 = modul.get_sql_history_price(coin2, connection, start_time, end_time)
        df = modul.make_spread_df(df_coin1, df_coin2, True, tf_5m)

        # Рассчитаем показатели статистики
        df = modul.zscore_calculating(df, 240)
        df['sma'] = talib.SMA(df.close, 240)
        df_coin1['sma'] = talib.SMA(df_coin1.close, 240)
        df_coin2['sma'] = talib.SMA(df_coin2.close, 240)

        pair_zscore = df.iloc[len(df) - 1]['zscore']
        pair_sma = df.iloc[len(df)-1]['sma']
        coin1_sma = df_coin1.iloc[len(df_coin1) - 1]['sma']
        coin2_sma = df_coin2.iloc[len(df_coin2) - 1]['sma']

        # получим последние цены
        is_index = coin1.find("indx")
        if is_index == -1:
            coin1_df = modul.get_last_price(coin1)
            last_price1 = coin1_df.iloc[0]['bid']
        else:
            # coin1_df = modul.get_last_index_price(coin1, connection)
            last_price1 = df_coin1.iloc[0]['close']
        coin2_df = modul.get_last_price(coin2)
        last_price2 = coin2_df.iloc[0]['bid']
        pair_price = last_price1 / last_price2

        # Рассчитаем нужные показатели

        pair_dev = (pair_price - pair_sma) / pair_sma * 100
        coin1_dev = (last_price1 - coin1_sma) / coin1_sma * 100
        coin2_dev = (last_price2 - coin2_sma) / coin2_sma * 100

        # Обновим данные в таблице sql
        modul.update_check_df(connection, pair, 'zscore', str(round(pair_zscore, 2)))
        modul.update_check_df(connection, pair, 'per_dev', str(round(pair_dev, 2)))
        modul.update_check_df(connection, pair, 'per_dev_c1', str(round(coin1_dev, 2)))
        modul.update_check_df(connection, pair, 'per_dev_c2', str(round(coin2_dev, 2)))
        modul.update_check_df(connection, pair, 'l_price', str(round(pair_price, 6)))


def find_new_pairs_for_bb3_art():

    atr3_filepath = r'.\screening\3_bb3_atr.csv'
    raw_filepath = r'.\screening\1_raw_result.csv'
    hard_filepath = r'.\screening\3_hard_check.csv'

    get_screening_result(False, 2000)
    # К сырым результатам добавим старые, но сначала добавим колонки монет по-отдельности.
    atr3_df = pd.read_csv(atr3_filepath, sep="\t")
    atr3_df[['coin1', 'coin2']] = atr3_df['pair'].str.split('-', 1, expand=True)
    atr3_df = atr3_df[['pair', 'coin1', 'coin2', 'corr', 'coint', 'stat1', 'stat2', 'stat_pair']]

    # Теперь добавим предыдущие результаты, т.к. они тоже могут пройти отбор
    raw_df = pd.read_csv(raw_filepath, sep="\t")
    new_df = pd.concat([raw_df, atr3_df], ignore_index=True)
    df_no_duplicates = new_df.drop_duplicates(subset=['pair'])
    df_no_duplicates.to_csv(raw_filepath, index=False, sep="\t")

    check_parameters_stability(window=1000, steps=4, shift=152, file=3, go_next=False)
    summarize_history_check()

    # Перезапишим готовые результаты в нужный файл
    hard_df = pd.read_csv(hard_filepath, sep="\t")
    hard_df.to_csv(atr3_filepath, index=False, sep="\t")


def find_new_pairs_for_grid1():
    get_screening_result(False, 9000)
    check_parameters_stability(window=4500, steps=4, shift=900, file=3, go_next=False)
    summarize_history_check()
    check_for_touch_bb(1000, 1, 2)


def looking_for_pump():
    # print(f'проверяем пары на наличие сильных движений')
    filepath_check = r'.\screening\1_raw_result.csv'
    end_time = datetime.datetime.now().timestamp()
    start_time = datetime.datetime.now().timestamp() - 500 * tf_5m
    if path.exists(filepath_check):
        check_df = pd.read_csv(filepath_check, sep="\t")
        for index in range(len(check_df)):
            coin1 = check_df.iloc[index]['coin1']
            coin2 = check_df.iloc[index]['coin2']
            pair = check_df.iloc[index]['pair']
            coin1_hist = modul.get_sql_history_price(coin1, connection, start_time, end_time)
            coin2_hist = modul.get_sql_history_price(coin2, connection, start_time, end_time)
            df = modul.make_spread_df(coin1_hist, coin2_hist, True, tf_5m)

            last_sma = df['close'].mean()
            _, _, price = modul.get_last_spread_price(coin1, coin2, connection)

            last_sma = float(last_sma)
            difference = round((price - last_sma)/last_sma*100, 2)
            if difference > 10.0 or difference < -10.0:
                if pair not in alerts:
                    print(f'{pair} изменение цены на {difference}%')
                    modul.send_message_to_telegram(f'{pair} изменение цены на {difference}%')
                    alerts.append(pair)
            elif pair in alerts:
                if -5.0 < difference < 5.0:
                    alerts.remove(pair)


if __name__ == '__main__':
    # ########### ПОРЯДОК ОТБОРА ПАР ###############
    # 1. Запускаем get_screening_result(True), результатом будет файл 1_raw_result.csv с первичным отбором
    # get_screening_result(True, 2000)

    # 2. затем идет проверка на стабильность, где проверяем отобранные в шаге 1 пары на 5ти периодах со смещением
    # в результате получаем файл 2_first_check.csv
    # check_parameters_stability(window=2000, steps=5, shift=72, file=3, go_next=True)

    # 3. Считаем среднее по рассчитанным на шаге 2 параметрам, убираем не подходящее условиям
    # на выходе файл 3_hard_check.csv
    # summarize_history_check(True)

    # 4. делаем грубую оптимизацию по отобранным в шаге 2 парам
    # грубую - то есть только по параметрам с sma=100. Потом сортируем по прибыльности, оставляем 50 лучших,
    # и по ним запускаем полную проверку. На выходе получаем файл 4_optimize_result.csv
    # get_historical_strategy_result(full_result=False, best_only=True)
    # get_historical_strategy_result(full_result=True, best_only=True)

    # 5. Проверка стационарности по торгуемым парам. получаем файл 5_actual_pairs_stability_check.csv
    # с 4мя периодами для проверки устойчивости параметров
    # check_parameters_stability(window=2000, steps=4, shift=144, file=1, go_next=True)

    # 6. Получаем среднее по шагу 5. записываем их в файл 7_final_result.csv (первое создание файла итогов)
    # отмечаем пары, у которых стационарность и коинтеграция хуже допустимых параметров
    # summarize_actual_pairs_stability(True)

    # 7. Делаем оптимизацию по уже торгуемым парам за последние 5-7(?) дней,
    # отмечаем 5 пар самым низким плановым доходом
    # создается файл 6_actual_pairs_optimize_result.csv
    # get_historical_strategy_result(full_result=True, best_only=True, file=1, days=7) # полный расчет, за 7 дней

    # 8. Последний. В файл 7_final_result добавляем данные из шага 7,
    # Потом из файла логов берем результаты торгов по каждой торгуемой паре за последние 5 дней,
    # рассчитываем доход за день, и так же добавляем в файл 7_final_result
    # calculate_trades_result()

    # 9. Ручной анализ - отмечаем 5-6 пар с самым низким среднедневным доходом, низким планом, низким фактом.
    # По отмеченным парам смотрим, где есть совпадения. Например, нет коинтеграции и самый низкий плановый доход,
    # или самый низкий плановый и фактический доход.
    # Принимаем решение об удалении пар из торгов, и замещении на новые из п.2.

    # #############################################################

    # блок отдельных процедур, для тестирования
    # end_time = datetime.datetime(2023, 4, 1, 0, 0, 0).timestamp()

    # параметры для большинства стратегий
    # get_screening_result(False, 2000, end_time)
    # get_screening_result(False, 2000)
    # check_parameters_stability(window=1000, steps=4, shift=152, file=3, go_next=False)
    # summarize_history_check()

    # find_new_pairs_for_bb3_art()

    # параметры для сеточных стратегий
    find_new_pairs_for_grid1()
    # check_for_touch_bb(1000, 1, 2)
    # get_screening_to_index(2000)

    # get_historical_strategy_result(False, True, 2) # 3_hard_check
    # get_historical_strategy_result(full_result=True, best_only=True, file=5, days=5) # полный расчет, xxx, за 5 дней
    # get_triangle_selected_list(3000)
    # get_bb_touch_list(1000)
    # check_for_touch_bb(1000, 1, 1)
    # fill_all_gaps(None, n=5)

    # #############################################################
    # Процедура поиска резких всплесков цены
    # schedule.every(5).minutes.do(looking_for_pump)
    # schedule.every(10).minutes.do(calculate_checkpairs_position)
    # while True:
    #     schedule.run_pending()
    #     time.sleep(180)

    # блок первичного заполнения исторических данных
    # start_t = datetime.datetime(2023, 2, 1, 0, 0).timestamp()
    # fill_deep_history_from(start_t)
    # fill_deep_sql_history_from(start_t)
    # check_parameters_stability(window=3000, steps=1, shift=300, file=3, go_next=False)
    # summarize_history_check()
    # cProfile.run("get_screening_result(False, 9000)")