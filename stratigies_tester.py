import pandas as pd
import bin_utils as modul
import talib
import datetime
from os import path
from pathlib import Path
import numpy as np
import BinScreener as scrn
import alor as alor_modul
import indicators as ind

tf_5m = 5 * 60
connection = modul.connect_to_sqlalchemy_binance()


def pivot_point_supertrend(spread_df, pp_prd, atr_factor, atr_prd):

    # Get high and low prices
    high = spread_df['high']
    low = spread_df['low']
    close = spread_df['close']
    # Calculate pivots
    pivot_high_series, pivot_low_series = ind.williams_fractals(spread_df, pp_prd)
    pivot_high_values = np.where(pivot_high_series, spread_df['high'], False)
    pivot_low_values = np.where(pivot_low_series, spread_df['low'], False)
    all_pivots = np.where(pivot_high_values, pivot_high_values, pivot_low_values)
    pd.Series(all_pivots).replace(0, np.nan, inplace=True)

    spread_df['pivots'] = all_pivots
    spread_df['center'] = np.nan
    # time_1_loop1 = datetime.datetime.now()
    for i in range((pp_prd+1), len(all_pivots)):
        pivot_now = all_pivots[i - pp_prd]
        if not pd.isna(pivot_now):
            if not pd.isna(spread_df.iloc[i-1]['center']):
                spread_df.at[i, 'center'] = (spread_df.iloc[i -1]['center'] * 2 + pivot_now) / 3
            else:
                spread_df.at[i, 'center'] = pivot_now
        else:
            if not pd.isna(spread_df.iloc[i - 1]['center']):
                spread_df.at[i, 'center'] = spread_df.iloc[i - 1]['center']
            else:
                spread_df.at[i, 'center'] = pivot_now

    # Upper/lower bands calculation
    atr = talib.ATR(high, low, close, timeperiod=atr_prd)
    up = spread_df['center'] - (atr_factor * atr)
    dn = spread_df['center'] + (atr_factor * atr)

    spread_df['trend'] = np.nan
    spread_df['trend_up'] = np.nan
    spread_df['trend_down'] = np.nan
    spread_df['switch'] = False
    spread_df['switch_to'] = ''
    # time_2_loop1 = datetime.datetime.now()
    for i in range(atr_prd,len(spread_df)):
        prev_trend = spread_df.iloc[i - 1]['trend']
        prev_trend_down = spread_df.iloc[i - 1]['trend_down']
        prev_trend_up = spread_df.iloc[i-1]['trend_up']
        prev_close = spread_df.iloc[i-1]['close']
        close = spread_df.iloc[i]['close']
        if pd.isna(prev_trend):
            if close < up[i]:
                spread_df.at[i, 'trend'] = dn[i]
                spread_df.at[i, 'switch_to'] = 'down'
                # ####################
                spread_df.at[i, 'trend_down'] = dn[i]
                spread_df.at[i, 'trend_up'] = up[i]
            else:
                spread_df.at[i, 'trend'] = up[i]
                spread_df.at[i, 'switch_to'] = 'up'
                # ####################
                spread_df.at[i, 'trend_down'] = dn[i]
                spread_df.at[i, 'trend_up'] = up[i]
            spread_df.at[i, 'switch'] = True

        else:
            # смотрим предыдущие значения, что бы понять, какой был тренд
            if prev_trend > prev_close:
                # значит был тренд вниз (down)
                if prev_trend >= close:
                    # пробития тренда не было, значит тренд остается
                    spread_df.at[i, 'trend'] = min(prev_trend, dn[i])
                    # ####################
                    spread_df.at[i, 'trend_down'] = min(prev_trend, dn[i])
                    if prev_trend_up > close:
                        spread_df.at[i, 'trend_up'] = up[i]
                    else:
                        spread_df.at[i, 'trend_up'] = max(prev_trend_up, up[i])
                else:
                    # тренд пробит, меняем линию на trend_up
                    spread_df.at[i, 'trend'] = max(prev_trend_up, up[i])
                    spread_df.at[i, 'switch'] = True
                    spread_df.at[i, 'switch_to'] = 'up'
                    # ####################
                    spread_df.at[i, 'trend_down'] = dn[i]  # не смысла смотреть min, точно знаем что цена пробила
                    spread_df.at[i, 'trend_up'] = max(prev_trend_up, up[i])

            elif prev_trend < prev_close:
                # значит был тренд вверх
                if prev_trend <= close:
                    # пробития тренда не было, значит тренд остается
                    spread_df.at[i, 'trend'] = max(prev_trend, up[i])
                    # ####################
                    spread_df.at[i, 'trend_up'] = max(prev_trend, up[i])
                    if prev_trend_down < close:
                        spread_df.at[i, 'trend_down'] = dn[i]
                    else:
                        spread_df.at[i, 'trend_down'] = min(prev_trend_down, dn[i])
                else:
                    # тренд пробит, меняем линию
                    spread_df.at[i, 'trend'] = min(prev_trend_down, dn[i])
                    spread_df.at[i, 'switch'] = True
                    spread_df.at[i, 'switch_to'] = 'down'
                    # ####################
                    spread_df.at[i, 'trend_up'] = up[i]  # не смысла смотреть max, точно знаем что цена пробила
                    spread_df.at[i, 'trend_down'] = min(prev_trend_down, dn[i])
            else:
                # Нужно учесть ситуацию цены равной тренду, в этом случае смены тренда еще не происходит
                # поэтому нужно посмотреть на две свечи назад. Бывает ли такое? Пока просто отмечу, что бы не захламлять код
                print("произошла исключительная ситуация, добавь исключение в код")
    # time_2_loop2 = datetime.datetime.now()
    # print(f'Second loop takes {(time_2_loop2 - time_2_loop1).seconds}')
    # print(f'All function takes {(time_2_loop2 - time_start).seconds}')
    return spread_df


def add_result(df, result, position, time):
    # создаем строку с данными
    new_row = pd.DataFrame({
        'position': [position],
        'result': [round(result, 2)],
        'time': [time],
    },
        index=None)
    df = pd.concat([df, new_row], ignore_index=True)
    return df


def add_new_position(signal, op_time, op_price, size):

    row = pd.DataFrame({
        'signal': [signal],
        'open_time': [op_time],
        'open_price': [round(op_price, 6)],
        'size': [round(size, 2)],
    }, index=None)

    return row


def close_new_position(df, signal, time, cl_price, direction, mae, mfe):
    df['cls_time'] = time
    df['cls_reason'] = signal
    df['cls_price'] = round(cl_price, 6)
    # df['mae'] = round(mae, 6)
    # df['mfe'] = round(mfe, 6)
    if direction == 'long':
        df['result_per'] = ((cl_price - df['open_price']) / df['open_price']) * 100
        # df['result'] = (cl_price - df['open_price']) * df['size']
        df['mae_per'] = np.where(df['open_price'] > df['cls_price'],
                                (df['cls_price'] - mae)/df['cls_price']*100,
                                (df['open_price'] - mae)/df['open_price']*100)
        df['mfe_per'] = np.where(df['open_price'] > df['cls_price'],
                                (mfe - df['open_price'])/df['open_price']*100,
                                (mfe - df['cls_price'])/df['cls_price']*100)
    else:
        df['result_per'] = ((df['open_price'] - cl_price) / cl_price) * 100
        # df['result'] = (df['open_price'] - cl_price) * df['size']
        df['mae_per'] = np.where(df['open_price'] > df['cls_price'],
                                (mae - df['open_price'])/df['open_price']*100,
                                (mae - df['cls_price'])/df['cls_price']*100)
        df['mfe_per'] = np.where(df['open_price'] > df['cls_price'],
                                (df['cls_price'] - mfe)/df['cls_price']*100,
                                (df['open_price'] - mfe)/df['open_price']*100)
    df['result_per'] = round(df['result_per'], 2)
    df['mae_per'] = round(df['mae_per'], 2)
    df['mfe_per'] = round(df['mfe_per'], 2)
    return df


def get_risk_correction(enter, stop, direction):
    """
    Рассчитывает коэф. коррекции размера позиции, в зависимости от размера риска.
    :param enter: цена входа
    :param stop: цена стоп лосса
    :param direction: направление (лонг/шорт)
    :return: float число.
    """
    plan_risk = 1.0
    if direction == 'long':
        potential_risk = (enter - stop) / enter * 100
        pos_correction = plan_risk / potential_risk
    else:
        potential_risk = (stop - enter) / enter * 100
        pos_correction = plan_risk / potential_risk

    return pos_correction


def get_selected_list(start_time, end_time, lookback):
    """
    Сканирует весь рынок на поиск пар,
    Сначала отбираются пары со стационарностью < 0.1, а потом подходящих под условия
    (4.0 < max_dev < 8.0) and (hl_dev < 3) and (all_counts > len(df)/100)
    :param start_time: дата С, timestamp
    :param end_time: Дата До, timestamp
    :return: датафрейм со всеми отобранными парами
    """
    res_df = pd.DataFrame()
    # получим список всех монет
    all_coins = modul.get_all_futures()
    all_futures = all_coins.id.tolist()
    all_futures2 = all_coins.id.tolist()
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
                        new_row = modul.get_statistics(future, future2, coin1_hist, coin2_hist, False)
                        if float(new_row.iloc[0]['stat_pair']) < 0.1:
                            df = modul.make_spread_df(coin1_hist, coin2_hist, last_to_end=True, tf=tf_5m)
                            dev_df = ind.get_max_deviation_from_sma(df, 1000)
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
                                # if (4.0 < max_dev < 9.0) and (hl_dev < 3) and (all_counts > len(df)/100): # отбор для класики
                                if 5.0 < max_dev < 10.0:  # отбор для выносов
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

    date_from_time = datetime.datetime.fromtimestamp(start_time).date()
    date_to_time = datetime.datetime.fromtimestamp(end_time).date()
    filename = f"scaning_result_{date_from_time}_{date_to_time}.csv"
    filepath = Path("testing", filename)
    res_df.to_csv(filepath, index=False, sep="\t")
    return res_df


def strategy_analyze(coin1, coin2, df_result):

    # Sharpe Ratio: This metric measures the risk-adjusted return of a strategy.
    # It compares the return of the strategy to the volatility of the returns.
    # The higher the Sharpe ratio, the better the risk-adjusted returns.
    #
    # Sortino Ratio: This metric is similar to the Sharpe ratio, but it only considers downside risk.
    # It measures the excess return of the strategy over a minimum acceptable return, divided by the downside risk.
    #
    # Average Win/Loss Ratio: This metric measures the average profit per winning trade versus the average loss
    # per losing trade. It can help you understand the consistency of the strategy's profits and losses.

    # посчитаем количество положительных и отрицательных сделок
    deals_count = len(df_result)
    if deals_count > 0:
        positive_df = df_result[df_result['result_per'] > 0]
        negative_df = df_result[df_result['result_per'] < 0]
        positive_count = len(positive_df)
        win_rate = positive_count/deals_count*100

        # avg_return = df_result['result_per'].mean()
        # std_dev = df_result['result_per'].std()
        # sharpe_ratio = np.sqrt(365) * (avg_return / std_dev)
        if len(negative_df) > 0:
            profit_factor = positive_df['result_per'].sum()/abs(negative_df['result_per'].sum())
        else:
            profit_factor = 100
        new_row = pd.DataFrame({
            'coin1': [coin1],
            'coin2': [coin2],
            'PnL': [df_result['result_per'].sum()],
            'drawdown': [df_result['drawdown'].min()],
            'deals': [deals_count],
            'MAE': [df_result['mae_per'].max()],
            'positive': [positive_count],
            'negative': [len(negative_df)],
            'win_rate': [round(win_rate, 2)],
            'profit_factor': [round(profit_factor, 2)],
        }, index=None)
    else:
        new_row = pd.DataFrame({
            'coin1': [coin1],
            'coin2': [coin2],
            'PnL': [0.0],
            'drawdown': [df_result['drawdown'].min()],
            'deals': [deals_count],
            'MAE': [df_result['mae_per'].max()],
            'positive': [0],
            'negative': [0],
            'win_rate': [0.0],
            'profit_factor': [0.0],
        }, index=None)
    return new_row


def check_list_for_strategies(start_time, end_time, file, lookback = 240):
    """
    Для проверки разных стратегий по одному и тому же списку монет, для их сравнения между собой.
    """
    if file == 1:
        filepath_check = r'.\reports\bin_to_check.csv'
    elif file == 2:
        filepath_check = r'.\screening\3_hard_check.csv'
    elif file == 3:
        filepath_check = r'.\screening\1_raw_result.csv'
    elif file == 4:
        filepath_check = r'.\screening\1_dist_result.csv'
    else:
        filepath_check = r'.\reports\xxx.csv'

    res_df = pd.DataFrame()
    full_df = pd.DataFrame()
    # end_time = datetime.datetime.now().timestamp()
    # start_time = end_time - 28000 * tf_5m

    if path.exists(filepath_check):
        check_df = pd.read_csv(filepath_check, sep="\t")

        for index in range(len(check_df)):
            if 'pair' in check_df.columns:
                pair = check_df.iloc[index]['pair']
                coin1, coin2 = modul.pair_to_coins(pair)
            else:
                coin1 = check_df.iloc[index]['coin1']
                coin2 = check_df.iloc[index]['coin2']
            print(f'проверяем пару {coin1}/{coin2}')
            # df = strategy_bb1_3_stop4(coin1, coin2, start_time, lookback, end_time)
            df = strategy_bb3atr_stop4(coin1, coin2, start_time, end_time, lookback)
            # df = strategy_grid_bb(coin1, coin2, start_time, end_time, lookback, 1, 5)
            # df = strategy_grid_bb_test(coin1, coin2, start_time, end_time, lookback, 1, 5)
            # df = strategy_pump_catcher(coin1, coin2, start_time, end_time, lookback, 0.1, 3)
            # df = strategy_dev1(coin1, coin2, start_time, end_time, lookback, 0.03)
            if len(df) > 0:
                new_row = strategy_analyze(coin1, coin2, df)
                res_df = pd.concat([res_df, new_row], ignore_index=True)
                df['pair'] = f'{coin1}-{coin2}'
                full_df = pd.concat([full_df, df], ignore_index=True)

    res_df.to_csv(r'.\reports\full_test_strategies.csv', index=False, sep="\t")
    full_df.sort_values(
        by='open_time',
        ascending=True,
        inplace=True,
        ignore_index=True,
    )
    full_df['cumulat_per'] = round(full_df['result_per'].cumsum(), 2)
    full_df['cum_max_per'] = round(full_df['cumulat_per'].cummax(), 2)
    full_df['drawdown'] = full_df['cumulat_per'] - full_df['cum_max_per']
    full_df.to_csv(r'.\reports\test_result.csv', index=False, sep="\t")


def walk_forward_scaning(start_time, end_time, scan_back, step_days, type='coint_div'):
    """
    Делает сканирование рынка за указанный период с шагом в одни сутки.
    От даты старта до даты окончания каждый день делаем поиск пар на
    заданную в scan_back глубину назад.

    :param start_time: дата С которой начинаем тестирование, в формате timestamp
    :param end_time: дата До которой тестируем, в формате timestamp
    :param scan_back: глубина сканирования
    :param step_days - количество дней для каждого сдвига
    :param type - выбор типа сканирования. only_coint, coint_div
    :return:
    """

    # Рассчитаем начало дня
    date_start = datetime.datetime.fromtimestamp(start_time)
    day_start = date_start.replace(hour=0, minute=0, second=0)
    # зададим исходный период для первого тестового окна
    from_time = datetime.datetime.timestamp(day_start)
    to_time = start_time + 1440*60*step_days

    while to_time <= end_time:
        scan_from = from_time - scan_back * tf_5m
        date_from_scan = datetime.datetime.fromtimestamp(scan_from).date()
        date_to_scan = datetime.datetime.fromtimestamp(from_time).date()
        if type == 'coint_div':
            filename_scan = f"scaning_result_{date_from_scan}_{date_to_scan}.csv"
            filepath_scan = Path("testing", filename_scan)
            if path.exists(filepath_scan):
                pd.read_csv(filepath_scan, sep="\t")
            else:
                # установим период отбора пар для торговли в тестовом окне
                get_selected_list(scan_from, from_time, scan_back)
                # pass
        else:
            filename_scan = f"coint_result_{date_from_scan}_{date_to_scan}.csv"
            filepath_scan = Path("testing", filename_scan)
            if path.exists(filepath_scan):
                pd.read_csv(filepath_scan, sep="\t")
            else:
                scrn.get_screening_result(False, scan_back, from_time)
                scrn.check_parameters_stability(window=int(scan_back/2), steps=4, shift=int(scan_back/10), file=4, go_next=False)
                scrn.summarize_history_check(False, 2)
                res_df = pd.read_csv(r'.\testing\3_history_hard_check.csv', sep="\t")
                res_df.to_csv(filepath_scan, index=False, sep="\t")
        from_time = to_time
        to_time = to_time + 1440 * 60 * step_days


def walk_forward_testing(start_time, end_time, scan_back, step_days, lookback, type='coint_div'):

    # Рассчитаем начало дня
    date_start = datetime.datetime.fromtimestamp(start_time)
    day_start = date_start.replace(hour=0, minute=0, second=0)
    # зададим исходный период для первого тестового окна
    from_time = datetime.datetime.timestamp(day_start)
    to_time = start_time + 1440 * 60 * step_days
    final_df = pd.DataFrame()
    trades_df = pd.DataFrame()
    while to_time <= end_time:
        res_df = pd.DataFrame()
        scan_from = from_time - scan_back * tf_5m
        date_from_scan = datetime.datetime.fromtimestamp(scan_from).date()
        date_to_scan = datetime.datetime.fromtimestamp(from_time).date()
        if type == 'coint_div':
            filename_scan = f"scaning_result_{date_from_scan}_{date_to_scan}.csv"
            filepath_scan = Path("testing", filename_scan)
            if path.exists(filepath_scan):
                check_df = pd.read_csv(filepath_scan, sep="\t")
            else:
                # установим период отбора пар для торговли в тестовом окне
                check_df = get_selected_list(scan_from, from_time, lookback)
        else:
            filename_scan = f"coint_result_{date_from_scan}_{date_to_scan}.csv"
            filepath_scan = Path("testing", filename_scan)
            if path.exists(filepath_scan):
                check_df = pd.read_csv(filepath_scan, sep="\t")
            else:
                # установим период отбора пар для торговли в тестовом окне
                scrn.get_screening_result(False, scan_back, from_time)
                scrn.check_parameters_stability(window=int(scan_back / 2), steps=4, shift=int(scan_back / 10), file=4,
                                                go_next=False)
                scrn.summarize_history_check(False, 2)
                check_df = pd.read_csv(r'.\testing\3_history_hard_check.csv', sep="\t")
                check_df.to_csv(filepath_scan, index=False, sep="\t")
        # Рассчитаем результат торговли по дням по полученным данным.
        for index in range(len(check_df)):
            if type == 'only_coint':
                pair = check_df.iloc[index]['pair']
                coin1, coin2 = modul.pair_to_coins(pair)
            else:
                coin1 = check_df.iloc[index]['coin1']
                coin2 = check_df.iloc[index]['coin2']
                bad_triangle = check_df.iloc[index]['bad_triangle']
                max_dev = check_df.iloc[index]['max_dev']
                if bad_triangle > 2 or max_dev > 10.0:
                    continue
            print(f'Расчет стратегии по паре {coin1}/{coin2}')
            to_time_with_addon = to_time+288*tf_5m*(step_days+3)  # добавляем 3 дня, чтобы закрыть сделки
            # df = strategy_bb1_3_stop4(coin1, coin2, from_time, lookback, to_time_with_addon)
            # df = test_best_bb_sigma(coin1, coin2, from_time, to_time_with_addon, lookback, 1)
            # df = strategy_bb3atr_stop4(coin1, coin2, from_time, to_time_with_addon, lookback)
            # df = strategy_grid_bb(coin1, coin2, from_time, to_time_with_addon, lookback, 1, 5)
            # df = strategy_grid_bb_test(coin1, coin2, from_time, to_time_with_addon, lookback, 1, 5)
            # df = strategy_grid_2_bb(coin1, coin2, from_time, to_time_with_addon, lookback, 1, 10)
            # df = strategy_dev1(coin1, coin2, from_time, to_time_with_addon, 500, 0.025)
            # df = strategy_zscore(coin1, coin2, from_time, to_time_with_addon, lookback, 3.0)
            # df = strategy_lregress_channel(coin1, coin2, from_time, to_time_with_addon, lookback, 2)
            df = strategy_structurebreak_catcher(coin1, coin2, from_time, to_time_with_addon, lookback, 1)

            # Распределим результаты по дням
            if len(df) > 0:
                work_df = df.loc[(df['open_time'] >= datetime.datetime.fromtimestamp(from_time + 15))
                                 & (df['open_time'] <= datetime.datetime.fromtimestamp(to_time))]
                work_df['pair'] = coin1 + '-' + coin2
                work_df['link'] = f'BINANCE:{coin1}.P/BINANCE:{coin2}.P'
                trades_df = pd.concat([trades_df, work_df], ignore_index=True)
                # Отключил расчет по дням, т.к. пока не использую
                # for day in range(step_days+2):
                #     filter_from = datetime.datetime.fromtimestamp(from_time + 288*tf_5m*day)
                #     filter_to = datetime.datetime.fromtimestamp(from_time + 288*tf_5m*(day+1))
                #     df_day = df.loc[(df['open_time'] >= filter_from) & (df['open_time'] <= filter_to)]
                #     if day == 0:
                #         new_row = strategy_analyze(coin1, coin2, df_day)
                #     else:
                #         next_row = strategy_analyze(coin1, coin2, df_day)
                #         new_row[f'PnL_{day}'] = next_row['PnL']
                #         new_row[f'drawdown_{day}'] = next_row['drawdown']
                #         new_row[f'deals_{day}'] = next_row['deals']
                #         new_row[f'MAE_{day}'] = next_row['MAE']
                #         new_row[f'posit_{day}'] = next_row['positive']
                #         new_row[f'negat_{day}'] = next_row['negative']
                #         new_row[f'win_r_{day}'] = next_row['win_rate']
                #         new_row[f'profit_f_{day}'] = next_row['profit_factor']

                # res_df = pd.concat([res_df, new_row], ignore_index=True)

        # final_df = pd.concat([final_df, res_df], ignore_index=True)

        # зададим период для следующего окна
        from_time = to_time
        to_time = to_time + 1440 * 60 * step_days

    # filename = f"strategy_final_grid.csv"
    # filepath = Path("testing", filename)
    # final_df.to_csv(filepath, index=False, sep="\t")

    trades_df['cumulat_per'] = round(trades_df['result_per'].cumsum(), 2)
    trades_df['cum_max_per'] = round(trades_df['cumulat_per'].cummax(), 2)
    trades_df['drawdown'] = trades_df['cumulat_per'] - trades_df['cum_max_per']

    filename2 = f"trades_grid.csv"
    filepath2 = Path("testing", filename2)
    trades_df.to_csv(filepath2, index=False, sep="\t")

    result_df = strategy_analyze('strategy', lookback, trades_df)
    filename3 = f"wft_result_period.csv"
    filepath3 = Path("testing", filename3)
    result_df.to_csv(filepath3, index=False, sep="\t")


def walk_forward_pump_portfolio_testing(start_time, end_time, scan_back, step_forward):
    # Варианты отбора для торговли пампов
    # - отбираем только по коинтеграции. Каждый день смотрим новые.
    # - сканируем весь рынок на поиск выносов... (тогда как отобрать пару для памповой монеты?)
    # - отбираем коинтегрированные, пополняя список каждый день (raw_result), не убирая, пока нет выноса.
    # после проторгованного выноса убираем.
    # - сканируем на поиск коинтегрированных пар с макс отклонением 10-15%

    pass


def single_strategy_testing(start_time, end_time):

    all_futures = modul.get_all_futures()
    # all_futures = ['1000LUNCUSDT', '1000PEPEUSDT', '1000SHIBUSDT', '1INCHUSDT', 'AAVEUSDT',
    #                'ADAUSDT', 'AGIXUSDT', 'ALGOUSDT', 'AMBUSDT', 'ANTUSDT', 'APEUSDT',
    #                'API3USDT', 'APTUSDT', 'ARBUSDT', 'ARPAUSDT', 'ARUSDT', 'ASTRUSDT', 'ATOMUSDT',
    #                'AVAXUSDT', 'AXSUSDT', 'BAKEUSDT', 'BANDUSDT', 'BCHUSDT',
    #                'BLZUSDT', 'BNBUSDT', 'BTCUSDT', 'C98USDT', 'CELOUSDT', 'CFXUSDT', 'CHZUSDT',
    #                'COMPUSDT', 'CRVUSDT', 'CYBERUSDT', 'DASHUSDT', 'DODOXUSDT', 'DOGEUSDT', 'DOTUSDT', 'DYDXUSDT',
    #                'EOSUSDT', 'ETCUSDT', 'FETUSDT', 'FILUSDT', 'FTMUSDT',
    #                'GALAUSDT', 'GALUSDT', 'GMTUSDT', 'GRTUSDT', 'GTCUSDT', 'HBARUSDT', 'ICPUSDT', 'IMXUSDT', 'INJUSDT',
    #                'KAVAUSDT', 'KNCUSDT', 'LDOUSDT', 'LEVERUSDT', 'LINAUSDT', 'LINKUSDT',
    #                'LPTUSDT', 'LTCUSDT', 'LUNA2USDT', 'MAGICUSDT', 'MANAUSDT', 'MASKUSDT', 'MATICUSDT',
    #                'MKRUSDT', 'MTLUSDT', 'NEARUSDT', 'OCEANUSDT',
    #                'OPUSDT', 'PENDLEUSDT', 'PEOPLEUSDT', 'RDNTUSDT', 'REEFUSDT', 'RNDRUSDT', 'RUNEUSDT',
    #                'SANDUSDT', 'SEIUSDT', 'SFPUSDT', 'SNXUSDT', 'SOLUSDT', 'STMXUSDT',
    #                'STXUSDT', 'SUIUSDT', 'SXPUSDT', 'THETAUSDT', 'TOMOUSDT', 'TRUUSDT', 'TRXUSDT',
    #                'UNFIUSDT', 'UNIUSDT', 'WAVESUSDT', 'WLDUSDT', 'WOOUSDT', 'XMRUSDT', 'XRPUSDT', 'XVGUSDT', 'YGGUSDT']

    trades_df = pd.DataFrame()
    for i in range(len(all_futures)):
        future = all_futures.iloc[i]["id"]
        # future = all_futures[i]
        print(f'Тестируем монету {future}')
        df = test_strategy_pp_supertrend(future, start_time, end_time, 2, 3, 10)
        # df = strategy_pp_supertrend_v2(future, start_time, end_time, 2, 3, 10)
        # df = strategy_pp_supertrend_v4(future, start_time, end_time, 2, 3, 10)
        if len(df) > 0:
            df['coin'] = future
            df['link'] = f'BINANCE:{future}.P'
            trades_df = pd.concat([trades_df, df], ignore_index=True)

    trades_df['cumulat_per'] = round(trades_df['result_per'].cumsum(), 2)
    trades_df['cum_max_per'] = round(trades_df['cumulat_per'].cummax(), 2)
    trades_df['drawdown'] = trades_df['cumulat_per'] - trades_df['cum_max_per']

    filename2 = f"single_trades.csv"
    filepath2 = Path("testing", filename2)
    trades_df.to_csv(filepath2, index=False, sep="\t")

    result_df = strategy_analyze('strategy', "", trades_df)
    filename3 = f"single_result_period.csv"
    filepath3 = Path("testing", filename3)
    result_df.to_csv(filepath3, index=False, sep="\t")


def single_strategy_testing_moex(start_time, end_time):
    headers = alor_modul.autorization()
    # list_of_sectors = ['FORTS', 'FOND', 'CURR']
    alor_connection = alor_modul.connect_to_sqlalchemy_moex()

    # for sector in list_of_sectors:
    #     list = alor_modul.fetch_securities_list(headers, sector)  # FORTS, FOND, CURR
    #     all_securites= list['symbol']
    # futures = ['CRU3', 'SIU3', 'EuU3', 'NGU3', 'BRU3', 'EDU3', 'GDU3', 'RIU3', 'MMU3', 'SVU3', 'GAZP', 'SBER', 'LKOH']  # экспир 9.23
    futures = ['CRM3', 'SIM3', 'EuM3', 'NGM3', 'BRM3', 'EDM3', 'GDM3', 'RIM3', 'MMM3', 'SVM3', 'GAZP', 'SBER', 'LKOH']  # экспир 6.23
    futures1 = ['CRH3', 'SIH3', 'EuH3', 'NGH3', 'BRH3', 'EDH3', 'GDH3', 'RIH3', 'MMH3', 'SVH3', 'GAZP', 'SBER', 'LKOH']  # экспир 3.23
    # shares = []
    trades_df = pd.DataFrame()
    for asset in futures:
        print(f'Тестируем инструмент {asset}')
        df = test_strategy_moex_pp_supertrend(asset, start_time, end_time, alor_connection, headers, 2, 3, 10)
        # df = strategy_pp_supertrend(future, start_time, end_time, 2, 3, 10)
        if len(df) > 0:
            df['coin'] = asset
            trades_df = pd.concat([trades_df, df], ignore_index=True)

    trades_df['cumulat_per'] = round(trades_df['result_per'].cumsum(), 2)
    trades_df['cum_max_per'] = round(trades_df['cumulat_per'].cummax(), 2)
    trades_df['drawdown'] = trades_df['cumulat_per'] - trades_df['cum_max_per']

    filename2 = f"single_trades_moex.csv"
    filepath2 = Path("testing", filename2)
    trades_df.to_csv(filepath2, index=False, sep="\t")

    result_df = strategy_analyze('strategy', "", trades_df)
    filename3 = f"single_result_period_moex.csv"
    filepath3 = Path("testing", filename3)
    result_df.to_csv(filepath3, index=False, sep="\t")


# ##############################################################
# Блок стратегий
#
# ##############################################################
def return_one_order_result(df_row, orders=1):
    """
    Процедура рассчитывает плановый результат по сделке.
    В строке таймфрейма должны передаваться данные по уже открытой сделке.
    Позволяет сравнить ручное вмешательство (факт по сделке) с автоторговлей
    :param df_row:
    :return:
    """

    coin1 = df_row['coin1']
    coin2 = df_row['coin2']
    op_price = df_row['op_price']
    op_time = pd.to_datetime(df_row['op_time'])
    strategy = df_row['strategy']
    lookback = df_row['lookback']
    going_to = df_row['going_to']
    # подготовим данные для расчета
    from_time = op_time.timestamp() - lookback*tf_5m - tf_5m
    to_time = op_time.timestamp() + lookback*3*tf_5m
    coin1_df = modul.get_sql_history_price(coin1, connection, from_time, to_time)
    coin2_df = modul.get_sql_history_price(coin2, connection, from_time, to_time)
    spread_df = modul.make_spread_df(coin1_df, coin2_df, True, tf_5m)
    if strategy == 'grid_1':
        spread_df['bb_up'], spread_df['sma'], spread_df['bb_down'] = talib.BBANDS(spread_df.close, lookback, 1, 1, 0)
        spread_df = spread_df[lookback:]
        spread_df.reset_index()
        for index in range(len(spread_df)):
            # вынем из дф нужные данные в переменные
            close = spread_df.iloc[index]['close']
            sma = spread_df.iloc[index]['sma']
            bb_up = spread_df.iloc[index]['bb_up']
            bb_down = spread_df.iloc[index]['bb_down']
            if going_to == 'DOWN':
                if orders > 2 and close < bb_down:
                    profit = (op_price - close) / op_price * 100
                    return round(profit, 3)
                elif orders <= 2 and close < sma:
                    profit = (op_price - close)/op_price*100
                    return round(profit, 3)
            else:
                if orders > 2 and close > bb_up:
                    profit = (close - op_price) / op_price * 100
                    return round(profit, 3)
                elif orders <= 2 and close > sma:
                    profit = (close - op_price) / op_price * 100
                    return round(profit, 3)
    elif strategy == 'bb3_atr':
        spread_df['bb_up'], spread_df['sma'], spread_df['bb_down'] = talib.BBANDS(spread_df.close, lookback, 4.2, 4.2, 0)
        spread_df = spread_df[lookback:]
        spread_df.reset_index()
        for index in range(len(spread_df)):
            # вынем из дф нужные данные в переменные
            close = spread_df.iloc[index]['close']
            sma = spread_df.iloc[index]['sma']
            bb_up = spread_df.iloc[index]['bb_up']
            bb_down = spread_df.iloc[index]['bb_down']
            if going_to == 'DOWN':
                if (close < sma) or (close > bb_up):
                    profit = (op_price - close) / op_price * 100
                    return round(profit, 3)
            else:
                if (close > sma) or (close < bb_down):
                    profit = (close - op_price) / op_price * 100
                    return round(profit, 3)
    # Если не дошло до точки закрытия, то возвращаем 0.0
    return 0.0


def test_oc_strategy(coin1, coin2, startdate, use_file):
    # Используется три линии ББ(период 240) 4.2, 3, и 1. Вход на возвращении цены в зону ББ-3,
    # Стоп - на уровень ББ-4 на момент открытия сделки.
    # Тейк тремя частями. Первая - на возврате в ББ-1 (в этот момент стоп передвигаем на ББ-3)
    # Второй - при пересечении SMA (стоп на ББ-1). Третий - на противоположный ББ-1

    # получение исходных данных
    if use_file:
        filepath_check = r'.\files\sber_sberp.csv'
        spread_df = pd.read_csv(filepath_check, sep=",")
        spread_df.drop(["bb1.dn", "bb1.up", "bb3.dn", "bb3.up", "bb4.dn", "bb4.up", "MA"], axis=1, inplace=True)
    else:
        # connection = modul.connect_to_sqlalchemy_binance()
        end = datetime.datetime.now().timestamp()
        df_coin1 = modul.get_sql_history_price(coin1, connection, startdate, end)
        df_coin2 = modul.get_sql_history_price(coin2, connection, startdate, end)
        spread_df = modul.make_spread_df(df_coin1, df_coin2, last_to_end=False, tf=tf_5m)

    spread_df['bb1_up'], spread_df['sma'], spread_df['bb1_down'] = talib.BBANDS(spread_df.close, 240, 1, 1, 0)
    spread_df['bb3_up'], _, spread_df['bb3_down'] = talib.BBANDS(spread_df.close, 240, 3, 3, 0)
    spread_df['bb4_up'], _, spread_df['bb4_down'] = talib.BBANDS(spread_df.close, 240, 4.2, 4.2, 0)

    # определелим переменные для расчетов
    result_df = pd.DataFrame(columns=['position', 'result', 'time'])
    in_position = False
    first_take_close = second_take_close = False
    enter_short = enter_long = 0.0
    stop_short = stop_long = 0.0
    wait_sma_cross = first_stop = False
    spread_df = spread_df.iloc[240:]
    for index in range(len(spread_df)):
        # вынем из дф нужные данные в переменные
        close = spread_df.iloc[index]['close']
        time = spread_df.iloc[index]['time']
        close_before = spread_df.iloc[index-1]['close']
        bb3_up = spread_df.iloc[index]['bb3_up']
        bb3_down = spread_df.iloc[index]['bb3_down']
        sma = spread_df.iloc[index]['sma']
        # Проверим, есть ли пересечение сма, если нам нужно его дождаться
        if wait_sma_cross:
            if (close_before < sma < close) | (close_before > sma > close):
                wait_sma_cross = False

        # сначала смотрим условия для открытия позиции
        if not in_position and not wait_sma_cross:
            if close_before > bb3_up > close:
                # открываем позицию в шорт
                enter_short = close
                stop_short = spread_df.iloc[index]['bb4_up']
                first_stop = True
                result_df = add_result(result_df, 0.0, 'open short', time)
                in_position = True
            elif close > bb3_down > close_before and close_before != 0:
                enter_long = close
                stop_long = spread_df.iloc[index]['bb4_down']
                first_stop = True
                result_df = add_result(result_df, 0.0, 'open long', time)
                in_position = True
        else:
            # сначала проверяем на превышение риска
            if close > stop_short and stop_short != 0.0:
                # закрываемся
                in_position = False
                result = (enter_short - close) / enter_short * 100
                result_df = add_result(result_df, result, 'stop loss', time)
                enter_short = stop_short = 0.0
                first_take_close = False
                second_take_close = False
                if first_stop:
                    wait_sma_cross = True
            elif close < stop_long:
                in_position = False
                result = (close - enter_long) / enter_long * 100
                result_df = add_result(result_df, result, 'stop loss', time)
                enter_long = stop_long = 0.0
                first_take_close = False
                second_take_close = False
                if first_stop:
                    wait_sma_cross = True

            # если не отстопило, проверяем на условия закрытия по тейку
            else:
                bb1_up = spread_df.iloc[index]['bb1_up']
                bb1_down = spread_df.iloc[index]['bb1_down']
                if first_take_close:
                    if second_take_close:
                        # смотрим последний уровень закрытия
                        if close < bb1_down and enter_short > 0.0:
                            # полностью закрываем шорт
                            in_position = False
                            result = ((enter_short - close) / enter_short * 100)/3
                            # result = 0.0
                            result_df = add_result(result_df, result, '3rd take', time)
                            enter_short = stop_short = 0.0
                            first_take_close = False
                            second_take_close = False
                            first_stop = False
                        elif close > bb1_up and enter_long > 0.0:
                            in_position = False
                            result = ((close - enter_long) / enter_long * 100)/3
                            # result = 0.0
                            result_df = add_result(result_df, result, '3rd take', time)
                            enter_long = stop_long = 0.0
                            first_take_close = False
                            second_take_close = False
                            first_stop = False
                    else:
                        if close < sma and enter_short > 0.0:
                            # берем треть от профита, т.к. закрыли бы только треть объема
                            result = ((enter_short - close) / enter_short * 100)/2
                            # result = 0.0
                            result_df = add_result(result_df, result, '2nd take', time)
                            stop_short = bb1_up
                            second_take_close = True
                            first_stop = False
                        elif close > sma and enter_long > 0.0:
                            result = ((close - enter_long) / enter_long * 100)/2
                            # result = 0.0
                            result_df = add_result(result_df, result, '2nd take', time)
                            stop_long = bb1_down
                            second_take_close = True
                            first_stop = False
                else:
                    if close < bb1_up and enter_short > 0.0:
                        # берем треть от профита, т.к. закрыли бы только треть объема
                        result = ((enter_short - close) / enter_short * 100) / 3
                        # result = 0.0
                        result_df = add_result(result_df, result, '1st take', time)
                        first_take_close = True
                        stop_short = bb3_up
                        first_stop = False
                    elif close > bb1_down and enter_long > 0.0:
                        result = ((close - enter_long) / enter_long * 100) / 3
                        # result = 0.0
                        result_df = add_result(result_df, result, '1st take', time)
                        first_take_close = True
                        stop_long = bb3_down
                        first_stop = False

    result_df.to_csv(r'.\reports\sber_sberp_result.csv', index=False, sep="\t")
    max_loss = result_df['result'].min()

    total = result_df['result'].sum()
    print(result_df)
    print(max_loss)
    print(total)


def test_oc_str_2takes(coin1, coin2, startdate, use_file):
    # Используется три линии ББ(период 240) 4.2, 3, и 1. Вход на возвращении цены в зону ББ-3,
    # Стоп - на уровень ББ-4 на момент открытия сделки.
    # Тейк двумя частями. Первая - на возврате в ББ-1 (в этот момент стоп передвигаем на цену открытия)
    # Вторая - при пересечении SMA.
    # При входе в сделку смотрим соотношение риска к плановому риску. Если превышет - снижаем сумму входа

    # получение исходных данных
    if use_file:
        filepath_check = r'.\files\sber_sberp.csv'
        spread_df = pd.read_csv(filepath_check, sep=",")
        spread_df.drop(["bb1.dn", "bb1.up", "bb3.dn", "bb3.up", "bb4.dn", "bb4.up", "MA"], axis=1, inplace=True)
    else:
        # connection = modul.connect_to_sqlalchemy_binance()
        end = datetime.datetime.now().timestamp()
        df_coin1 = modul.get_sql_history_price(coin1, connection, startdate, end)
        df_coin2 = modul.get_sql_history_price(coin2, connection, startdate, end)
        spread_df = modul.make_spread_df(df_coin1, df_coin2, last_to_end=True, tf=tf_5m)

    spread_df['bb1_up'], spread_df['sma'], spread_df['bb1_down'] = talib.BBANDS(spread_df.close, 240, 1, 1, 0)
    spread_df['bb3_up'], aaa, spread_df['bb3_down'] = talib.BBANDS(spread_df.close, 240, 3, 3, 0)
    spread_df['bb4_up'], bbb, spread_df['bb4_down'] = talib.BBANDS(spread_df.close, 240, 4.2, 4.2, 0)

    # определелим переменные для расчетов
    result_df = pd.DataFrame(columns=['position', 'result', 'time'])
    in_position = False
    first_take_close = False
    # second_take_close = False
    enter_short = enter_long = 0.0
    stop_short = stop_long = 0.0
    # plan_risk = 1.0
    pos_correction = 0.0
    wait_sma_cross = first_stop = False
    spread_df = spread_df.iloc[240:]
    for index in range(len(spread_df)):
        # вынем из дф нужные данные в переменные
        close = spread_df.iloc[index]['close']
        time = spread_df.iloc[index]['startTime']
        close_before = spread_df.iloc[index-1]['close']
        bb3_up = spread_df.iloc[index]['bb3_up']
        bb3_down = spread_df.iloc[index]['bb3_down']
        bb4_up = spread_df.iloc[index]['bb4_up']
        bb4_down = spread_df.iloc[index]['bb4_down']
        sma = spread_df.iloc[index]['sma']
        # Проверим, есть ли пересечение сма, если нам нужно его дождаться
        if wait_sma_cross:
            if (close_before < sma < close) | (close_before > sma > close):
                wait_sma_cross = False

        # сначала смотрим условия для открытия позиции
        if not in_position and not wait_sma_cross:
            if close_before > bb3_up > close:
                potential = (close - sma) / close * 100
                if potential > 0.4:
                    # pos_correction = get_risk_correction(close, bb4_up, 'short')
                    pos_correction = 1.0
                    # открываем позицию в шорт
                    enter_short = close
                    stop_short = bb4_up
                    first_stop = True
                    result_df = add_result(result_df, 0.0, 'open short', time)
                    in_position = True
            elif close > bb3_down > close_before and close_before != 0:
                potential = (sma - close) / close * 100
                if potential > 0.4:
                    # pos_correction = get_risk_correction(close, bb4_down, 'long')
                    pos_correction = 1.0
                    enter_long = close
                    stop_long = bb4_down
                    first_stop = True
                    result_df = add_result(result_df, 0.0, 'open long', time)
                    in_position = True
        else:
            # сначала проверяем на условие стоп лосса
            if close > stop_short and stop_short != 0.0:
                # закрываемся
                in_position = False
                result = ((enter_short - close) / enter_short * 100) * pos_correction
                if first_take_close:
                    result = result/2
                result_df = add_result(result_df, result, 'stop loss', time)
                enter_short = stop_short = pos_correction = 0.0
                first_take_close = False
                # second_take_close = False
                if first_stop:
                    wait_sma_cross = True
            elif close < stop_long:
                in_position = False
                result = ((close - enter_long) / enter_long * 100) * pos_correction
                if first_take_close:
                    result = result/2
                result_df = add_result(result_df, result, 'stop loss', time)
                enter_long = stop_long = pos_correction = 0.0
                first_take_close = False
                # second_take_close = False
                if first_stop:
                    wait_sma_cross = True

            # если не отстопило, проверяем на условия закрытия по тейку
            else:
                bb1_up = spread_df.iloc[index]['bb1_up']
                bb1_down = spread_df.iloc[index]['bb1_down']
                if first_take_close:
                    if close < sma and enter_short > 0.0:
                        # берем треть от профита, т.к. закрыли бы только треть объема
                        result = (((enter_short - close) / enter_short * 100)/2) * pos_correction
                        # result = 0.0
                        result_df = add_result(result_df, result, '2nd take', time)
                        in_position = False
                        enter_short = stop_short = pos_correction = 0.0
                        first_take_close = False
                        # second_take_close = True
                        first_stop = False
                    elif close > sma and enter_long > 0.0:
                        result = (((close - enter_long) / enter_long * 100)/2) * pos_correction
                        # result = 0.0
                        result_df = add_result(result_df, result, '2nd take', time)
                        in_position = False
                        enter_long = stop_long = pos_correction = 0.0
                        first_take_close = False
                        # second_take_close = True
                        first_stop = False
                else:
                    if close < bb1_up and enter_short > 0.0:
                        # берем треть от профита, т.к. закрыли бы только треть объема
                        result = (((enter_short - close) / enter_short * 100) / 2) * pos_correction
                        # result = 0.0
                        result_df = add_result(result_df, result, '1st take', time)
                        first_take_close = True
                        stop_short = enter_short
                        first_stop = False
                    elif close > bb1_down and enter_long > 0.0:
                        result = (((close - enter_long) / enter_long * 100) / 2) * pos_correction
                        # result = 0.0
                        result_df = add_result(result_df, result, '1st take', time)
                        first_take_close = True
                        stop_long = enter_long
                        first_stop = False

    result_df.to_csv(r'.\reports\test_result.csv', index=False, sep="\t")
    max_loss = result_df['result'].min()

    total = result_df['result'].sum()
    print(result_df)
    print(max_loss)
    print(total)


def test_best_bb_sigma(coin1, coin2, start_date, end_date, lookback, sigma):
    # Используется три линии ББ(период 240) 4.2, 3, и 1. Вход на возвращении цены в зону ББ-3,
    # Стоп - на уровень ББ-4 на момент открытия сделки.
    # Тейк двумя частями. Первая - на возврате в ББ-1 (в этот момент стоп передвигаем на цену открытия)
    # Вторая - при пересечении SMA.
    # При входе в сделку смотрим соотношение риска к плановому риску. Если превышет - снижаем сумму входа

    #
    # v.1 Вход при выходе цены за линию ББ, закрытие на SMA. стопов нет.
    # Результат лучше на сигме=1, по мере увеличения сигмы pnl падает на 20-30%
    #
    # v.2 Вход на возврате внутрь канала ББ. Закрытие на SMA.
    # Существенных изменений не принесло. Доходность чуть снизилась, просадка тоже незначительно.

    start_date = start_date - lookback * tf_5m  # для того, что бы расчет стратегии начался с правильных показаний индик.
    df_coin1 = modul.get_sql_history_price(coin1, connection, start_date, end_date)
    df_coin2 = modul.get_sql_history_price(coin2, connection, start_date, end_date)
    spread_df = modul.make_spread_df(df_coin1, df_coin2, last_to_end=True, tf=tf_5m)

    # spread_df['bb1_up'], spread_df['sma'], spread_df['bb1_down'] = talib.BBANDS(spread_df.close, 240, 1, 1, 0)
    spread_df['bb_up'], spread_df['sma'], spread_df['bb_down'] = talib.BBANDS(spread_df.close, lookback, sigma, sigma, 0)
    # spread_df['bb4_up'], bbb, spread_df['bb4_down'] = talib.BBANDS(spread_df.close, 240, 4.2, 4.2, 0)
    spread_df = spread_df.iloc[lookback:]

    # определелим переменные для расчетов
    enter_short = enter_long = 0.0
    wait_sma_cross = False

    result_df = df = pd.DataFrame()
    in_position = False
    mae = mfe = 0.0
    amount = 250.0

    for index in range(len(spread_df)):
        # вынем из дф нужные данные в переменные
        close = spread_df.iloc[index]['close']
        time = spread_df.iloc[index]['startTime']
        close_before = spread_df.iloc[index-1]['close']
        bb_up = spread_df.iloc[index]['bb_up']
        bb_down = spread_df.iloc[index]['bb_down']
        sma = spread_df.iloc[index]['sma']
        size = amount / close

        # сначала смотрим условия для открытия позиции
        if not in_position:
            # Проверяем на условие первого входа
            if close > bb_up > close_before:
                potential = (bb_up - sma) / bb_up * 100
                if potential > 0.5:
                    # открываем позицию в шорт
                    in_position = True
                    df = add_new_position('bb short', time, bb_up, size)
                    mae = mfe = enter_short = bb_up

            elif close_before < close < bb_down:
                potential = (sma - bb_down) / bb_down * 100
                if potential > 0.5:
                    # открываем позицию в long
                    in_position = True
                    df = add_new_position('bb long', time, bb_down, size)
                    mae = mfe = enter_long = bb_down

        else:
            if close < sma and enter_short > 0.0:
                # Закрываем все позиции
                in_position = False
                if close < mfe:
                    mfe = close
                df = close_new_position(df, 'take', time, close, 'short', mae, mfe)
                result_df = pd.concat([result_df, df], ignore_index=True)
                enter_short = 0.0
                mae = mfe = 0.0
            elif close > sma and enter_long > 0.0:
                in_position = False
                if close > mfe:
                    mfe = close
                df = close_new_position(df, 'take', time, close, 'long', mae, mfe)
                result_df = pd.concat([result_df, df], ignore_index=True)
                enter_long = 0.0
                mae = mfe = 0.0

    if in_position:
        if enter_short > 0.0:
            df = close_new_position(df, 'time', time, close, 'long', mae, mfe)
            result_df = pd.concat([result_df, df], ignore_index=True)
        else:
            df = close_new_position(df, 'time', time, close, 'short', mae, mfe)
            result_df = pd.concat([result_df, df], ignore_index=True)

    if len(result_df) > 0:
        result_df['cumulat_per'] = result_df['result_per'].cumsum()
        result_df['cum_max_per'] = result_df['cumulat_per'].cummax()
        result_df['drawdown'] = result_df['cumulat_per'] - result_df['cum_max_per']

        result_df.to_csv(r'.\reports\test_result.csv', index=False, sep="\t")
        total = result_df['result_per'].sum()
        drawdown = result_df['drawdown'].min()
        print(f'Drawdown = {drawdown}')
        print(f'Total PnL = {total}')
    return result_df


def strategy_bb1_3_stop4(coin1, coin2, start_date, lookback, end_date=None):
    # Используется три линии ББ(период 240) 4.2, 3, и 1. Вход на возвращении цены в зону ББ-1,
    # Второй вход при возврате в зону ББ-3. Если Цена сразу ушла в ББ-3, то на ББ-1 уже не входим.
    # Стоп - если цена пересекает линию ББ-4. Или если сделка длится дольше HalfLife
    # После срабатывания стопа ждем пересечения SMA.
    # Тейк при пересечении SMA. Не входим при потенциале сделки менее 0.5%

    start_date = start_date - lookback * tf_5m  # для того, что бы расчет стратегии начался с правильных показаний индик.
    # получение исходных данных
    if end_date is None:
        end_date = datetime.datetime.now().timestamp()
    df_coin1 = modul.get_sql_history_price(coin1, connection, start_date, end_date)
    df_coin2 = modul.get_sql_history_price(coin2, connection, start_date, end_date)
    spread_df = modul.make_spread_df(df_coin1, df_coin2, last_to_end=True, tf=tf_5m)

    spread_df['bb1_up'], spread_df['sma'], spread_df['bb1_down'] = talib.BBANDS(spread_df.close, lookback, 1, 1, 0)
    spread_df['bb3_up'], aaa, spread_df['bb3_down'] = talib.BBANDS(spread_df.close, lookback, 3, 3, 0)
    spread_df['bb4_up'], bbb, spread_df['bb4_down'] = talib.BBANDS(spread_df.close, lookback, 4.2, 4.2, 0)

    # определелим переменные для расчетов
    result_df = df = pd.DataFrame()
    in_position = wait_sma_cross = False
    half_life = lookback/2
    enter_short = enter_long = 0.0
    enter_short3 = enter_long3 = 0.0
    bar_count = bars_before_open = 0
    mae = mfe = 0.0
    amount = 100.0
    spread_df = spread_df.iloc[lookback:]
    for index in range(len(spread_df)):
        # вынем из дф нужные данные в переменные
        close = spread_df.iloc[index]['close']
        time = spread_df.iloc[index]['startTime']
        if index > 0:
            close_before = spread_df.iloc[index-1]['close']
        else:
            close_before = spread_df.iloc[index]['close']
        size = amount/close
        bb3_up = spread_df.iloc[index]['bb3_up']
        bb3_down = spread_df.iloc[index]['bb3_down']
        bb4_up = spread_df.iloc[index]['bb4_up']
        bb4_down = spread_df.iloc[index]['bb4_down']
        bb1_up = spread_df.iloc[index]['bb1_up']
        bb1_down = spread_df.iloc[index]['bb1_down']
        sma = spread_df.iloc[index]['sma']

        if (close_before < sma < close) | (close_before > sma > close):
            bars_before_open = 0  # Начинаем отсчет свечей от момента пересечения SMA
            # Проверим, есть ли пересечение сма, если нам нужно его дождаться
            if wait_sma_cross:
                wait_sma_cross = False
        elif not wait_sma_cross:
            bars_before_open += 1
            # сначала смотрим условия для открытия позиции
            if not in_position:
                if close_before > bb1_up > close:
                    potential = (bb1_up - sma) / bb1_up * 100
                    if potential > 0.5:
                        # открываем позицию в шорт
                        enter_short = bb1_up
                        in_position = True
                        bar_count = 0
                        df = add_new_position('bb-1 short', time, bb1_up, size)
                        mae = mfe = bb1_up
                elif close > bb1_down > close_before:
                    potential = (sma - bb1_down) / bb1_down * 100
                    if potential > 0.5:
                        enter_long = bb1_down
                        in_position = True
                        bar_count = 0
                        df = add_new_position('bb-1 long', time, bb1_down, size)
                        mae = mfe = bb1_down
                # Если цена сразу пролетела до ББ-3, то сразу открываем сделку на возврате в ББ-3
                elif close_before > bb3_up > close:
                    potential = (bb3_up - sma) / bb3_up * 100
                    if potential > 0.5:
                        # открываем позицию в шорт
                        enter_short = enter_short3 = bb3_up
                        in_position = True
                        bar_count = 0
                        df = add_new_position('bb-3 short', time, bb3_up, size)
                        mae = mfe = bb3_up
                elif close > bb3_down > close_before:
                    potential = (sma - bb3_down) / bb3_down * 100
                    if potential > 0.5:
                        enter_long = enter_long3 = bb3_down
                        in_position = True
                        bar_count = 0
                        df = add_new_position('bb-3 long', time, bb3_down, size)
                        mae = mfe = bb3_down
            else:
                bar_count += 1
                # сначала проверяем на условие стоп лосса
                if close > bb4_up and enter_short > 0.0:
                    # закрываемся
                    in_position = False
                    wait_sma_cross = True
                    enter_short = enter_short3 = 0.0
                    if close < mfe:
                        mfe = close
                    df = close_new_position(df, 'stop bb-4', time, bb4_up, 'short', mae, mfe)
                    result_df = pd.concat([result_df, df], ignore_index=True)
                elif close < bb4_down and enter_long > 0.0:
                    in_position = False
                    wait_sma_cross = True
                    enter_long = enter_long3 = 0.0
                    df = close_new_position(df, 'stop bb-4', time, bb4_down, 'long', mae, mfe)
                    result_df = pd.concat([result_df, df], ignore_index=True)

                # Проверим нет ли стопа по времени
                elif bar_count > half_life:
                    in_position = False
                    wait_sma_cross = True
                    if enter_short > 0.0:
                        df = close_new_position(df, 'half life', time, close, 'short', mae, mfe)
                        result_df = pd.concat([result_df, df], ignore_index=True)
                        enter_short = enter_short3 = 0.0
                    elif enter_long > 0.0:
                        df = close_new_position(df, 'half life', time, close, 'long', mae, mfe)
                        result_df = pd.concat([result_df, df], ignore_index=True)
                        enter_long = enter_long3 = 0.0

                # если не отстопило, проверяем на условия закрытия по тейку
                else:
                    # Возможно цена в уровне открытия ББ-3
                    if (close_before > bb3_up > close) and enter_short > 0.0 and enter_short3 == 0.0:
                        potential = (bb3_up - sma) / bb3_up * 100
                        if potential > 0.5:
                            # открываем позицию в шорт
                            enter_short3 = bb3_up
                            in_position = True
                            new_row3 = add_new_position('bb-3 short', time, bb3_up, size)
                            df = pd.concat([df, new_row3], ignore_index=True)
                    elif (close > bb3_down > close_before) and enter_long > 0.0 and enter_long3 == 0.0:
                        potential = (sma - bb3_down) / bb3_down * 100
                        if potential > 0.5:
                            enter_long3 = bb3_down
                            in_position = True
                            new_row3 = add_new_position('bb-3 long', time, bb3_down, size)
                            df = pd.concat([df, new_row3], ignore_index=True)

                    # Проверим не пора ли закрывать позицию по тейку
                    elif close < sma and enter_short > 0.0:
                        in_position = False
                        enter_short = enter_short3 = 0.0
                        df = close_new_position(df, 'take', time, sma, 'short', mae, mfe)
                        result_df = pd.concat([result_df, df], ignore_index=True)

                    elif close > sma and enter_long > 0.0:
                        in_position = False
                        enter_long = enter_long3 = 0.0
                        df = close_new_position(df, 'take', time, sma, 'long', mae, mfe)
                        result_df = pd.concat([result_df, df], ignore_index=True)
                    elif (close > mae and enter_short > 0.0) | (close < mae and enter_long > 0.0):
                        mae = close
                    elif (close < mfe and enter_short > 0.0) | (close > mfe and enter_long > 0.0):
                        mfe = close

    if len(result_df) > 0:
        result_df['cumulat_per'] = round(result_df['result_per'].cumsum(), 2)
        result_df['cum_max_per'] = round(result_df['cumulat_per'].cummax(), 2)
        result_df['drawdown'] = result_df['cumulat_per'] - result_df['cum_max_per']

        result_df.to_csv(r'.\reports\test_result.csv', index=False, sep="\t")
        total = result_df['result_per'].sum()
        drawdown = result_df['drawdown'].min()
        print(f'Drawdown = {drawdown}')
        print(f'Total PnL = {total}')
    return result_df


def strategy_bb3atr_stop4(coin1, coin2, start_date, end_date, lookback):
    # Используется две линии ББ 4.2 и 3. Вход на возвращении цены в зону ББ-3,
    # при условии что ATR больше среднего в 3 раза.
    # Стоп - если цена пересекает линию ББ-4. Или если сделка длится дольше HalfLife
    # После срабатывания стопа ждем пересечения SMA.
    # Тейк при пересечении SMA. Не входим при потенциале сделки менее 0.5%
    # тест1 - со стопом на bb4 при открытии позиции стабильно убыточен
    # тест2 - стоп в безубыток при достижении ценой половины пути до sma - pnl хуже на 10%, просадка такая же

    start_date = start_date - lookback * tf_5m  # для того, что бы расчет стратегии начался с правильных показаний индик.
    # получение исходных данных
    df_coin1 = modul.get_sql_history_price(coin1, connection, start_date, end_date)
    df_coin2 = modul.get_sql_history_price(coin2, connection, start_date, end_date)
    spread_df = modul.make_spread_df(df_coin1, df_coin2, last_to_end=True, tf=tf_5m)

    spread_df['bb3_up'], spread_df['sma'], spread_df['bb3_down'] = talib.BBANDS(spread_df.close, lookback, 3, 3, 0)
    spread_df['bb4_up'], bbb, spread_df['bb4_down'] = talib.BBANDS(spread_df.close, lookback, 4.2, 4.2, 0)
    # Calculate ATR with a period of 14
    spread_df['atr'] = talib.ATR(spread_df['high'], spread_df['low'], spread_df['close'], timeperiod=5)
    spread_df['sma_atr'] = spread_df['atr'].rolling(window=lookback, min_periods=1).mean()
    # Calculate ATR percentage
    spread_df['atr_percentage'] = (spread_df['atr'] / spread_df['close']) * 100
    # определелим переменные для расчетов
    result_df = df = pd.DataFrame()
    in_position = wait_sma_cross = False
    half_life = lookback/2
    enter_short = enter_long = 0.0
    bar_count = bars_before_open = 0
    mae = mfe = 0.0
    stop = 0.0
    amount = 250.0
    spread_df = spread_df.iloc[lookback:]
    for index in range(len(spread_df)):
        # вынем из дф нужные данные в переменные
        close = spread_df.iloc[index]['close']
        time = spread_df.iloc[index]['startTime']
        if index > 0:
            close_before = spread_df.iloc[index-1]['close']
            if index > 5:
                atr_df = spread_df[index-5:index]
                atr_max = atr_df['atr'].max()
                atr_coeff = atr_max / spread_df.iloc[index]['sma_atr']
            else:
                atr_coeff = spread_df.iloc[index - 1]['atr'] / spread_df.iloc[index]['sma_atr']
        else:
            close_before = spread_df.iloc[index]['close']
            atr_coeff = spread_df.iloc[index]['atr'] / spread_df.iloc[index]['sma_atr']
        size = amount/close
        bb3_up = spread_df.iloc[index]['bb3_up']
        bb3_down = spread_df.iloc[index]['bb3_down']
        bb4_up = spread_df.iloc[index]['bb4_up']
        bb4_down = spread_df.iloc[index]['bb4_down']
        sma = spread_df.iloc[index]['sma']

        if (close_before < sma < close) | (close_before > sma > close):
            bars_before_open = 0  # Начинаем отсчет свечей от момента пересечения SMA
            # Проверим, есть ли пересечение сма, если нам нужно его дождаться
            if wait_sma_cross:
                wait_sma_cross = False
        elif not wait_sma_cross:
            bars_before_open += 1
            # сначала смотрим условия для открытия позиции
            if not in_position:
                if close_before > bb3_up > close and atr_coeff > 2.0:
                    potential = (bb3_up - sma) / bb3_up * 100
                    if potential > 0.5:
                        # открываем позицию в шорт
                        enter_short = bb3_up
                        in_position = True
                        bar_count = 0
                        stop = 0.0
                        df = add_new_position('bb-3 short', time, bb3_up, size)
                        mae = mfe = bb3_up
                elif close > bb3_down > close_before and atr_coeff > 2.0:
                    potential = (sma - bb3_down) / bb3_down * 100
                    if potential > 0.5:
                        enter_long = bb3_down
                        in_position = True
                        bar_count = 0
                        stop = 0.0
                        df = add_new_position('bb-3 long', time, bb3_down, size)
                        mae = mfe = bb3_down
            else:
                bar_count += 1
                # сначала проверяем на условие стоп лосса
                if close > bb4_up and enter_short > 0.0:
                    # закрываемся
                    in_position = False
                    wait_sma_cross = True
                    enter_short = stop = 0.0
                    if close < mfe:
                        mfe = close
                    df = close_new_position(df, 'stop bb-4', time, bb4_up, 'short', mae, mfe)
                    result_df = pd.concat([result_df, df], ignore_index=True)
                elif close < bb4_down and enter_long > 0.0:
                    in_position = False
                    wait_sma_cross = True
                    enter_long = stop = 0.0
                    df = close_new_position(df, 'stop bb-4', time, bb4_down, 'long', mae, mfe)
                    result_df = pd.concat([result_df, df], ignore_index=True)
                # Проверим нет ли стопа по времени
                elif bar_count > half_life:
                    in_position = False
                    wait_sma_cross = True
                    if enter_short > 0.0:
                        df = close_new_position(df, 'half life', time, close, 'short', mae, mfe)
                        result_df = pd.concat([result_df, df], ignore_index=True)
                        enter_short = stop = 0.0
                    elif enter_long > 0.0:
                        df = close_new_position(df, 'half life', time, close, 'long', mae, mfe)
                        result_df = pd.concat([result_df, df], ignore_index=True)
                        enter_long = stop = 0.0
                # # проверим стоп в безубыток
                # elif stop != 0.0 and close > stop and enter_short > 0.0:
                #     df = close_new_position(df, 'stop no loss', time, close, 'short', mae, mfe)
                #     result_df = pd.concat([result_df, df], ignore_index=True)
                #     enter_short = stop = 0.0
                # elif stop != 0.0 and close < stop and enter_long > 0.0:
                #     df = close_new_position(df, 'stop no loss', time, close, 'long', mae, mfe)
                #     result_df = pd.concat([result_df, df], ignore_index=True)
                #     enter_long = stop = 0.0
                # если не отстопило, проверяем на условия закрытия по тейку
                else:
                    # Проверим не пора ли закрывать позицию по тейку
                    if close < sma and enter_short > 0.0:
                        in_position = False
                        enter_short = stop = 0.0
                        df = close_new_position(df, 'take', time, sma, 'short', mae, mfe)
                        result_df = pd.concat([result_df, df], ignore_index=True)
                    elif close > sma and enter_long > 0.0:
                        in_position = False
                        enter_long = stop = 0.0
                        df = close_new_position(df, 'take', time, sma, 'long', mae, mfe)
                        result_df = pd.concat([result_df, df], ignore_index=True)
                    elif (close > mae and enter_short > 0.0) | (close < mae and enter_long > 0.0):
                        mae = close
                    elif (close < mfe and enter_short > 0.0) | (close > mfe and enter_long > 0.0):
                        mfe = close
                    # if stop == 0.0:
                    #     # ставим стоп в безубыток, если ушли в профит на пол расстояния до sma
                    #     op_price = df.iloc[0]['open_price']
                    #     if enter_short > 0.0:
                    #         all_way = op_price - sma
                    #         covered_way = op_price - close
                    #         if covered_way > all_way/2:
                    #             stop = op_price
                    #     elif enter_long > 0.0:
                    #         all_way = sma - op_price
                    #         covered_way = close - op_price
                    #         if covered_way > all_way/2:
                    #             stop = op_price

    if in_position:
        if enter_short > 0.0:
            df = close_new_position(df, 'time', time, close, 'short', mae, mfe)
            result_df = pd.concat([result_df, df], ignore_index=True)
        else:
            df = close_new_position(df, 'time', time, close, 'long', mae, mfe)
            result_df = pd.concat([result_df, df], ignore_index=True)

    if len(result_df) > 0:
        result_df['cumulat_per'] = round(result_df['result_per'].cumsum(), 2)
        result_df['cum_max_per'] = round(result_df['cumulat_per'].cummax(), 2)
        result_df['drawdown'] = result_df['cumulat_per'] - result_df['cum_max_per']

        result_df.to_csv(r'.\reports\test_result.csv', index=False, sep="\t")
        total = result_df['result_per'].sum()
        drawdown = result_df['drawdown'].min()
        print(f'Drawdown = {drawdown}')
        print(f'Total PnL = {total}')
    return result_df


def strategy_pump_catcher(coin1, coin2, start_date, end_date, lookback, percent, grid_rows):
    # Ловим выносы от SMA больше чем на percent
    # Следующие входы на расстоянии step от цены входа.
    # Тейк - при пересечении SMA.
    # варианты тестов
    # - меняем период сма, percent. step, rows...
    # - пробуем тянуть вход до разворота цены. (по пред свече? по % отката? по пересечению быстр sma?)
    # - фильтруем входы по скорости роста цены. (берем только быстрые)

    start_date = start_date - lookback * tf_5m  # для того, что бы расчет стратегии начался с правильных показаний индик.
    df_coin1 = modul.get_sql_history_price(coin1, connection, start_date, end_date)
    df_coin2 = modul.get_sql_history_price(coin2, connection, start_date, end_date)
    spread_df = modul.make_spread_df(df_coin1, df_coin2, last_to_end=True, tf=tf_5m)
    spread_df['sma'] = spread_df['close'].rolling(window=lookback, min_periods=1).mean()

    result_df = df = pd.DataFrame()
    in_position = False
    last_short = last_long = 0.0
    current_level = 1
    step = 0.02
    mae = mfe = next_level = 0.0
    amount = 100.0
    spread_df = spread_df.iloc[lookback:]
    for index in range(len(spread_df)):
        # вынем из дф нужные данные в переменные
        close = spread_df.iloc[index]['close']
        time = spread_df.iloc[index]['startTime']
        sma = spread_df.iloc[index]['sma']
        lvl_up = sma + sma*percent
        lvl_down = sma - sma * percent
        size = amount / close
        if not in_position:
            # Проверяем на условие первого входа
            if close > lvl_up:
                # открываем позицию в шорт
                last_short = lvl_up
                in_position = True
                next_level = last_short + last_short * step
                df = add_new_position('lvl-1 short', time, lvl_up, size/grid_rows)
                mae = mfe = lvl_up

            elif close < lvl_down:
                # открываем позицию в long
                last_long = lvl_down
                in_position = True
                next_level = last_long - last_long * step
                df = add_new_position('lvl-1 long', time, lvl_down, size/grid_rows)
                mae = mfe = lvl_down
        else:
            # Если не все уровни открыты, смотрим не пора ли открыть новый
            if last_short > 0.0:
                if close > next_level and current_level <= grid_rows:
                    last_short = close
                    next_level = last_short + (last_short * step)
                    new_row2 = add_new_position(f'short grid-{current_level}', time, close, size/grid_rows)
                    df = pd.concat([df, new_row2], ignore_index=True)
                    current_level += 1
                elif close < sma:
                    # Закрываем все позиции
                    current_level = 1
                    in_position = False
                    if close < mfe:
                        mfe = close
                    df = close_new_position(df, 'take', time, close, 'short', mae, mfe)
                    result_df = pd.concat([result_df, df], ignore_index=True)
                    last_short = next_level = 0.0
                    mae = mfe = 0.0
                elif close > mae:
                    mae = close
                elif close < mfe:
                    mfe = close
            else:
                if close < next_level and current_level <= grid_rows:
                    last_long = close
                    next_level = last_long - last_long * step
                    new_row2 = add_new_position(f'long grid-{current_level}', time, close, size/grid_rows)
                    df = pd.concat([df, new_row2], ignore_index=True)
                    current_level += 1
                elif close > sma:
                    # Закрываем все позиции
                    current_level = 1
                    in_position = False
                    if close > mfe:
                        mfe = close
                    df = close_new_position(df, 'take', time, close, 'long', mae, mfe)
                    result_df = pd.concat([result_df, df], ignore_index=True)
                    last_long = next_level = 0.0
                    mae = mfe = 0.0
                elif close < mae:
                    mae = close
                elif close > mfe:
                    mfe = close

    if in_position:
        if last_short > 0.0:
            df = close_new_position(df, 'time', time, close, 'short', mae, mfe)
            result_df = pd.concat([result_df, df], ignore_index=True)
        else:
            df = close_new_position(df, 'time', time, close, 'long', mae, mfe)
            result_df = pd.concat([result_df, df], ignore_index=True)

    if len(result_df) > 0:
        result_df['cumulat_per'] = result_df['result_per'].cumsum()
        result_df['cum_max_per'] = result_df['cumulat_per'].cummax()
        result_df['drawdown'] = result_df['cumulat_per'] - result_df['cum_max_per']

        result_df.to_csv(r'.\reports\test_result.csv', index=False, sep="\t")
        total = result_df['result_per'].sum()
        drawdown = result_df['drawdown'].min()
        print(f'Drawdown = {drawdown}')
        print(f'Total PnL = {total}')
    return result_df


def strategy_grid_bb(coin1, coin2, start_date, end_date, lookback, sigma, grid_rows):
    # Используется одна линия ББ(период 1000) с первым отклонением. Вход при пересечении ББ-1, наружу
    # Следующие входы на расстоянии Х% от цены входа. Не входим при потенциале сделки менее 0.5%
    # Тейк - если был только один вход, то при пересечении SMA. Если больше - то на противоположном ББ-1

    start_date = start_date - lookback*2*tf_5m #для того, что бы расчет стратегии начался с правильных показаний индик.
    df_coin1 = modul.get_sql_history_price(coin1, connection, start_date, end_date)
    df_coin2 = modul.get_sql_history_price(coin2, connection, start_date, end_date)
    spread_df = modul.make_spread_df(df_coin1, df_coin2, last_to_end=True, tf=tf_5m)
    result_df = df = pd.DataFrame()

    spread_df['bb_up'], spread_df['sma'], spread_df['bb_down'] = talib.BBANDS(spread_df.close, lookback, sigma, sigma, 0)
    in_position = False
    last_short = last_long = 0.0
    current_level = 1
    step = 0.02
    mae = mfe = next_level = 0.0
    amount = 250.0
    check_df = spread_df.copy()
    # проверим, что в отбор не попали заведомо не пригодные пары
    _, _, time_to_opposite = modul.check_for_touch_bb(check_df[:lookback * 2], lookback, sigma)
    if time_to_opposite > lookback / 2:
        return result_df

    spread_df = spread_df.iloc[lookback*2:]
    for index in range(len(spread_df)):
        # вынем из дф нужные данные в переменные
        close = spread_df.iloc[index]['close']
        time = spread_df.iloc[index]['startTime']
        bb_up = spread_df.iloc[index]['bb_up']
        bb_down = spread_df.iloc[index]['bb_down']
        sma = spread_df.iloc[index]['sma']
        size = amount / close
        if not in_position:
            # Проверяем на условие первого входа
            if close > bb_up:
                potential = (bb_up - sma) / bb_up * 100
                if potential > 0.5:
                    # открываем позицию в шорт
                    last_short = bb_up
                    in_position = True
                    next_level = last_short + last_short * step
                    df = add_new_position('bb-1 short', time, bb_up, size/grid_rows)
                    mae = mfe = bb_up

            elif close < bb_down:
                potential = (sma - bb_down) / bb_down * 100
                if potential > 0.5:
                    # открываем позицию в long
                    last_long = bb_down
                    in_position = True
                    next_level = last_long - last_long * step
                    df = add_new_position('bb-1 long', time, bb_down, size/grid_rows)
                    mae = mfe = bb_down
        else:
            # Если не все уровни открыты, смотрим не пора ли открыть новый
            if last_short > 0.0:
                if close > next_level and current_level <= grid_rows:
                    last_short = close
                    next_level = last_short + (last_short * step)
                    new_row2 = add_new_position(f'short grid-{current_level}', time, close, size/grid_rows)
                    df = pd.concat([df, new_row2], ignore_index=True)
                    current_level += 1
                elif close < sma and current_level == 1:
                    # Закрываем все позиции
                    current_level = 1
                    in_position = False
                    if close < mfe:
                        mfe = close
                    df = close_new_position(df, 'take', time, close, 'short', mae, mfe)
                    result_df = pd.concat([result_df, df], ignore_index=True)
                    last_short = next_level = 0.0
                    mae = mfe = 0.0
                elif close < bb_down and current_level > 1:
                    # Закрываем все позиции
                    current_level = 1
                    in_position = False
                    if close < mfe:
                        mfe = close
                    df = close_new_position(df, 'take', time, close, 'short', mae, mfe)
                    result_df = pd.concat([result_df, df], ignore_index=True)
                    last_short = next_level = 0.0
                    mae = mfe = 0.0
                elif close > mae:
                    mae = close
                elif close < mfe:
                    mfe = close
            else:
                if close < next_level and current_level <= grid_rows:
                    last_long = close
                    next_level = last_long - last_long * step
                    new_row2 = add_new_position(f'long grid-{current_level}', time, close, size/grid_rows)
                    df = pd.concat([df, new_row2], ignore_index=True)
                    current_level += 1
                elif close > sma and current_level == 1:
                    # Закрываем все позиции
                    current_level = 1
                    in_position = False
                    if close > mfe:
                        mfe = close
                    df = close_new_position(df, 'take', time, close, 'long', mae, mfe)
                    result_df = pd.concat([result_df, df], ignore_index=True)
                    last_long = next_level = 0.0
                    mae = mfe = 0.0
                elif close > bb_up and current_level > 1:
                    # Закрываем все позиции
                    current_level = 1
                    in_position = False
                    if close > mfe:
                        mfe = close
                    df = close_new_position(df, 'take', time, close, 'long', mae, mfe)
                    result_df = pd.concat([result_df, df], ignore_index=True)
                    last_long = next_level = 0.0
                    mae = mfe = 0.0
                elif close < mae:
                    mae = close
                elif close > mfe:
                    mfe = close
    if in_position:
        if last_short > 0.0:
            df = close_new_position(df, 'time', time, close, 'short', mae, mfe)
            result_df = pd.concat([result_df, df], ignore_index=True)
        else:
            df = close_new_position(df, 'time', time, close, 'long', mae, mfe)
            result_df = pd.concat([result_df, df], ignore_index=True)

    if len(result_df) > 0:
        result_df['cumulat_per'] = result_df['result_per'].cumsum()
        result_df['cum_max_per'] = result_df['cumulat_per'].cummax()
        result_df['drawdown'] = result_df['cumulat_per'] - result_df['cum_max_per']

        result_df.to_csv(r'.\reports\test_result.csv', index=False, sep="\t")
        total = result_df['result_per'].sum()
        drawdown = result_df['drawdown'].min()
        print(f'Drawdown = {drawdown}')
        print(f'Total PnL = {total}')
    return result_df


def strategy_grid_2_bb(coin1, coin2, start_date, end_date, lookback, sigma, grid_rows):
    # Используется одна линия ББ(период 1000) с первым отклонением. Вход при пересечении ББ-1, наружу
    # Следующие входы на расстоянии Х% от цены входа. Выход - при достижении уровня step*2
    # Использовать с большой сетью ордеров (10-20).

    start_date = start_date - lookback*2*tf_5m #для того, что бы расчет стратегии начался с правильных показаний индик.
    df_coin1 = modul.get_sql_history_price(coin1, connection, start_date, end_date)
    df_coin2 = modul.get_sql_history_price(coin2, connection, start_date, end_date)
    spread_df = modul.make_spread_df(df_coin1, df_coin2, last_to_end=True, tf=tf_5m)
    result_df = df = pd.DataFrame()

    spread_df['bb_up'], spread_df['sma'], spread_df['bb_down'] = talib.BBANDS(spread_df.close, lookback, sigma, sigma, 0)
    in_position = False
    last_short = last_long = 0.0
    current_level = 1
    step = 0.01
    mae = mfe = next_level = 0.0
    amount = 250.0
    check_df = spread_df.copy()
    # проверим, что в отбор не попали заведомо не пригодные пары
    _, _, time_to_opposite = modul.check_for_touch_bb(check_df[:lookback * 2], lookback, sigma)
    if time_to_opposite > lookback / 2:
        return result_df

    spread_df = spread_df.iloc[lookback*2:]
    for index in range(len(spread_df)):
        # вынем из дф нужные данные в переменные
        close = spread_df.iloc[index]['close']
        time = spread_df.iloc[index]['startTime']
        bb_up = spread_df.iloc[index]['bb_up']
        bb_down = spread_df.iloc[index]['bb_down']
        sma = spread_df.iloc[index]['sma']
        size = amount / close
        if not in_position:
            # Проверяем на условие первого входа
            if close > bb_up:
                potential = (bb_up - sma) / bb_up * 100
                if potential > 0.5:
                    # открываем позицию в шорт
                    last_short = bb_up
                    in_position = True
                    next_level = last_short + last_short * step
                    df = add_new_position('bb-1 short', time, bb_up, size/grid_rows)
                    mae = mfe = bb_up

            elif close < bb_down:
                potential = (sma - bb_down) / bb_down * 100
                if potential > 0.5:
                    # открываем позицию в long
                    last_long = bb_down
                    in_position = True
                    next_level = last_long - last_long * step
                    df = add_new_position('bb-1 long', time, bb_down, size/grid_rows)
                    mae = mfe = bb_down
        else:
            # Если не все уровни открыты, смотрим не пора ли открыть новый
            if last_short > 0.0:
                close_level = close + close*step*2
                df_for_close = df[df['open_price'] > close_level]
                deals_for_close = df_for_close.shape[0]
                if close > next_level and current_level <= grid_rows:
                    last_short = close
                    next_level = last_short + (last_short * step)
                    new_row2 = add_new_position(f'short grid-{current_level}', time, close, size/grid_rows)
                    df = pd.concat([df, new_row2], ignore_index=True)
                    current_level += 1
                elif deals_for_close > 0:
                    # Закрываем часть позиций
                    if len(df) > deals_for_close:
                        current_level = current_level - deals_for_close
                        df = df[:len(df)-deals_for_close] # Убираем позиции для закрытия из списка остальных сделок
                        closed_df = close_new_position(df_for_close, 'take', time, close, 'short', mae, mfe)
                        result_df = pd.concat([result_df, closed_df], ignore_index=True)
                        last_short = df.iloc[len(df)-1]['open_price']
                        next_level = last_short + (last_short * step)
                    else: #Значит закрываем все позиции
                        current_level = 1
                        in_position = False
                        df = close_new_position(df, 'take', time, close, 'short', mae, mfe)
                        result_df = pd.concat([result_df, df], ignore_index=True)
                        last_short = next_level = 0.0
                        mae = mfe = 0.0
                elif close < bb_down:
                    # Закрываем все позиции
                    current_level = 1
                    in_position = False
                    if close < mfe:
                        mfe = close
                    df = close_new_position(df, 'take', time, close, 'short', mae, mfe)
                    result_df = pd.concat([result_df, df], ignore_index=True)
                    last_short = next_level = 0.0
                    mae = mfe = 0.0
                elif close > mae:
                    mae = close
                elif close < mfe:
                    mfe = close
            else:
                close_level = close - close * step * 2
                df_for_close = df[df['open_price'] < close_level]
                deals_for_close = df_for_close.shape[0]
                if close < next_level and current_level <= grid_rows:
                    last_long = close
                    next_level = last_long - last_long * step
                    new_row2 = add_new_position(f'long grid-{current_level}', time, close, size/grid_rows)
                    df = pd.concat([df, new_row2], ignore_index=True)
                    current_level += 1
                elif deals_for_close > 0:
                    # Закрываем часть позиций
                    if len(df) > deals_for_close:
                        current_level = current_level - deals_for_close
                        df = df[:len(df)-deals_for_close]  # Убираем позиции для закрытия из списка остальных сделок
                        closed_df = close_new_position(df_for_close, 'take', time, close, 'long', mae, mfe)
                        result_df = pd.concat([result_df, closed_df], ignore_index=True)
                        last_long = df.iloc[len(df) - 1]['open_price']
                        next_level = last_long - last_long * step
                    else:  # Значит закрываем все позиции
                        current_level = 1
                        in_position = False
                        if close > mfe:
                            mfe = close
                        df = close_new_position(df, 'take', time, close, 'long', mae, mfe)
                        result_df = pd.concat([result_df, df], ignore_index=True)
                        last_long = next_level = 0.0
                        mae = mfe = 0.0
                elif close > bb_up and current_level > 1:
                    # Закрываем все позиции
                    current_level = 1
                    in_position = False
                    if close > mfe:
                        mfe = close
                    df = close_new_position(df, 'take', time, close, 'long', mae, mfe)
                    result_df = pd.concat([result_df, df], ignore_index=True)
                    last_long = next_level = 0.0
                    mae = mfe = 0.0
                elif close < mae:
                    mae = close
                elif close > mfe:
                    mfe = close
    if in_position:
        if last_short > 0.0:
            df = close_new_position(df, 'time', time, close, 'short', mae, mfe)
            result_df = pd.concat([result_df, df], ignore_index=True)
        else:
            df = close_new_position(df, 'time', time, close, 'long', mae, mfe)
            result_df = pd.concat([result_df, df], ignore_index=True)

    if len(result_df) > 0:
        result_df['cumulat_per'] = result_df['result_per'].cumsum()
        result_df['cum_max_per'] = result_df['cumulat_per'].cummax()
        result_df['drawdown'] = result_df['cumulat_per'] - result_df['cum_max_per']

        result_df.to_csv(r'.\reports\test_result.csv', index=False, sep="\t")
        total = result_df['result_per'].sum()
        drawdown = result_df['drawdown'].min()
        print(f'Drawdown = {drawdown}')
        print(f'Total PnL = {total}')
    return result_df


def strategy_grid_bb_test(coin1, coin2, start_date, end_date, lookback, sigma, grid_rows):
    # Доработки стратегии Grid_bb
    # v1. Увеличиваем расстояние между уровнями (step + current_level/100)
    # По тестам результат хуже, чем у исходной на 20-30%
    #
    # v2. Меняем тейки. все кроем на SMA.
    # По тестам PnL хуже, чем у исходной на 30%. Но (!) MAE лучше на 10%
    #
    # v.3 Все уровни кроме первого кроем на возврате в ББ. Первый кроем на SMA.
    # PnL в 4(!!!) раза хуже. остальные параметры нет смысла даже смотреть.
    #
    # v.4 Открываем первую позицию не на выходе, а на возврате в ББ
    # Совокупно на 10% хуже, но много пар показали лучший pnl чем у исходной.
    # Просадка хуже на 1%, а MAE лучше(!) на 1%
    #
    # v.5 Дополнительные сделки закрываем при достижении профита в step или при пересечении SMA
    # Лучший результат показало закрытие на step*2, но даже так в среднем на 10% pnl хуже.
    # И сильно хуже (в 4-5 раз) просадка. Доходит до 40-50%
    #
    # v.6 Отсеиваем пары на входе, у которых нет касания одной из линий ББ более чем lookback/2
    # Т.е. за полпериода цена должна коснуться противоположных сторон ББ минимум 1 раз.
    # Результат PnL хуже в среднем на 20-30%, но(!!!) но примерно на столько же лучше просадка.
    #
    # v.7 Ищем стоп-лосс. Пробуем стопиться когда цена ушла больше чем на 10%
    # Результаты в таблице. Однозначных нет. При включении фильтра отсева пар по касаниям линий ББ
    # результаты падают вдвое, но просадка снижается до 10-20%!!!
    #
    # v.8 - v.6 + закрытие по SMA с уменьшающимся периодом после lookback/2.
    # Т.е. если при нахождении в сделке не происходит закрытия в течении lookback/2, то с каждой следующей
    # свечей уменьшаем lookback на 1.

    start_date = start_date - lookback * 2 * tf_5m  # для того, что бы расчет стратегии начался с правильных показаний индик.
    df_coin1 = modul.get_sql_history_price(coin1, connection, start_date, end_date)
    df_coin2 = modul.get_sql_history_price(coin2, connection, start_date, end_date)
    spread_df = modul.make_spread_df(df_coin1, df_coin2, last_to_end=True, tf=tf_5m)
    result_df = df = pd.DataFrame()

    spread_df['bb_up'], spread_df['sma'], spread_df['bb_down'] = talib.BBANDS(spread_df.close, lookback, sigma, sigma, 0)
    in_position = False
    last_short = last_long = 0.0
    current_level = 1
    step = 0.02
    mae = mfe = next_level = 0.0
    amount = 250.0
    stop = 10.0
    wait_sma_cross = False
    check_df = spread_df.copy()
    # проверим, что в отбор не попали заведомо не пригодные пары
    # _, _, time_to_opposite = modul.check_for_touch_bb(check_df[:lookback * 2], lookback, sigma)
    # if time_to_opposite > lookback / 2:
    #     return result_df
    spread_df = spread_df.iloc[lookback*2:]
    for index in range(len(spread_df)):
        # _, _, time_to_opposite = modul.check_for_touch_bb(check_df[lookback:(lookback*2+index)], lookback, sigma)
        # if time_to_opposite > lookback / 2 and not in_position:
        #    continue

        # вынем из дф нужные данные в переменные
        if index > 0:
            close_before = spread_df.iloc[index-1]['close']
        else:
            close_before = spread_df.iloc[index]['close']
        close = spread_df.iloc[index]['close']
        time = spread_df.iloc[index]['startTime']
        bb_up = spread_df.iloc[index]['bb_up']
        bb_down = spread_df.iloc[index]['bb_down']
        sma = spread_df.iloc[index]['sma']
        size = amount / close

        if wait_sma_cross:
            if (close_before < sma < close) | (close_before > sma > close):
                # Проверим, есть ли пересечение сма, если нам нужно его дождаться
                wait_sma_cross = False
        else:
            # сначала смотрим условия для открытия позиции
            if not in_position:
                # Проверяем на условие первого входа
                if close > bb_up:
                    potential = (bb_up - sma) / bb_up * 100
                    if potential > 0.5:
                        # открываем позицию в шорт
                        last_short = bb_up
                        in_position = True
                        next_level = last_short + last_short * step
                        df = add_new_position('bb-1 short', time, bb_up, size / grid_rows)
                        mae = mfe = bb_up

                elif close < bb_down:
                    potential = (sma - bb_down) / bb_down * 100
                    if potential > 0.5:
                        # открываем позицию в long
                        last_long = bb_down
                        in_position = True
                        next_level = last_long - last_long * step
                        df = add_new_position('bb-1 long', time, bb_down, size / grid_rows)
                        mae = mfe = bb_down
            else:
                # Если не все уровни открыты, смотрим не пора ли открыть новый
                if last_short > 0.0:
                    if close > next_level and current_level <= grid_rows:
                        last_short = close
                        next_level = last_short + (last_short * step)
                        new_row2 = add_new_position(f'short grid-{current_level}', time, close, size / grid_rows)
                        df = pd.concat([df, new_row2], ignore_index=True)
                        current_level += 1
                    elif close < sma and current_level == 1:
                        # Закрываем все позиции
                        current_level = 1
                        in_position = False
                        if close < mfe:
                            mfe = close
                        df = close_new_position(df, 'take', time, close, 'short', mae, mfe)
                        result_df = pd.concat([result_df, df], ignore_index=True)
                        last_short = next_level = 0.0
                        mae = mfe = 0.0
                    elif close < bb_down and current_level > 1:
                        # Закрываем все позиции
                        current_level = 1
                        in_position = False
                        if close < mfe:
                            mfe = close
                        df = close_new_position(df, 'take', time, close, 'short', mae, mfe)
                        result_df = pd.concat([result_df, df], ignore_index=True)
                        last_short = next_level = 0.0
                        mae = mfe = 0.0
                    elif (close - sma)/sma*100 > stop:
                        # Закрываем по стопу
                        current_level = 1
                        in_position = False
                        wait_sma_cross = True
                        if close < mfe:
                            mfe = close
                        df = close_new_position(df, 'stop', time, close, 'short', mae, mfe)
                        result_df = pd.concat([result_df, df], ignore_index=True)
                        last_short = next_level = 0.0
                        mae = mfe = 0.0
                    elif close > mae:
                        mae = close
                    elif close < mfe:
                        mfe = close
                else:
                    if close < next_level and current_level <= grid_rows:
                        last_long = close
                        next_level = last_long - last_long * step
                        new_row2 = add_new_position(f'long grid-{current_level}', time, close, size / grid_rows)
                        df = pd.concat([df, new_row2], ignore_index=True)
                        current_level += 1
                    elif close > sma and current_level == 1:
                        # Закрываем все позиции
                        current_level = 1
                        in_position = False
                        if close > mfe:
                            mfe = close
                        df = close_new_position(df, 'take', time, close, 'long', mae, mfe)
                        result_df = pd.concat([result_df, df], ignore_index=True)
                        last_long = next_level = 0.0
                        mae = mfe = 0.0
                    elif close > bb_up and current_level > 1:
                        # Закрываем все позиции
                        current_level = 1
                        in_position = False
                        if close > mfe:
                            mfe = close
                        df = close_new_position(df, 'take', time, close, 'long', mae, mfe)
                        result_df = pd.concat([result_df, df], ignore_index=True)
                        last_long = next_level = 0.0
                        mae = mfe = 0.0
                    elif (sma-close)/sma*100 > stop:
                        # Закрываем по стопу
                        current_level = 1
                        in_position = False
                        wait_sma_cross = True
                        if close > mfe:
                            mfe = close
                        df = close_new_position(df, 'stop', time, close, 'long', mae, mfe)
                        result_df = pd.concat([result_df, df], ignore_index=True)
                        last_long = next_level = 0.0
                        mae = mfe = 0.0
                    elif close < mae:
                        mae = close
                    elif close > mfe:
                        mfe = close
    if in_position:
        if last_short > 0.0:
            df = close_new_position(df, 'time', time, close, 'short', mae, mfe)
            result_df = pd.concat([result_df, df], ignore_index=True)
        else:
            df = close_new_position(df, 'time', time, close, 'long', mae, mfe)
            result_df = pd.concat([result_df, df], ignore_index=True)

    if len(result_df) > 0:
        result_df['cumulat_per'] = result_df['result_per'].cumsum()
        result_df['cum_max_per'] = result_df['cumulat_per'].cummax()
        result_df['drawdown'] = result_df['cumulat_per'] - result_df['cum_max_per']

        result_df.to_csv(r'.\reports\test_result.csv', index=False, sep="\t")
        total = result_df['result_per'].sum()
        drawdown = result_df['drawdown'].min()
        print(f'Drawdown = {drawdown}')
        print(f'Total PnL = {total}')
    return result_df


def strategy_dev1(coin1, coin2, start_date, end_date, lookback, step_line):
    # Используется SMA и две линии на Х процентов от SMA,
    # Вход при выходе за линию (не ждем close), в направлении к SMA,
    # и сразу же вход в противоположную сторону. Выходим на противоположной стороне, либо
    # по достижении уровня противоположной стороны на момент входа.
    # Стоп - если уровень входа пересекается SMA

    start_date = start_date - lookback * tf_5m  # для того, что бы расчет стратегии начался с правильных показаний индик.
    # получение исходных данных
    if end_date is None:
        end_date = datetime.datetime.now().timestamp()
    df_coin1 = modul.get_sql_history_price(coin1, connection, start_date, end_date)
    df_coin2 = modul.get_sql_history_price(coin2, connection, start_date, end_date)
    spread_df = modul.make_spread_df(df_coin1, df_coin2, last_to_end=True, tf=tf_5m)

    spread_df['sma'] = spread_df['close'].rolling(window=lookback, min_periods=1).mean()
    spread_df['line_up'] = spread_df['sma'] + spread_df['sma']*step_line
    spread_df['line_down'] = spread_df['sma'] - spread_df['sma']*step_line

    # определелим переменные для расчетов
    result_df = df = pd.DataFrame()
    in_position = False
    enter_short = enter_long = 0.0
    take_long = take_short = 0
    mae = mfe = 0.0
    amount = 100.0
    spread_df = spread_df.iloc[lookback:]
    for index in range(len(spread_df)):
        # вынем из дф нужные данные в переменные
        close = spread_df.iloc[index]['close']
        high = spread_df.iloc[index]['high']
        low = spread_df.iloc[index]['low']
        time = spread_df.iloc[index]['startTime']
        # if index > 0:
        #     close_before = spread_df.iloc[index-1]['close']
        # else:
        #     close_before = spread_df.iloc[index]['close']
        size = amount/close
        line_up = spread_df.iloc[index]['line_up']
        line_down = spread_df.iloc[index]['line_down']
        sma = spread_df.iloc[index]['sma']

        # сначала смотрим условия для открытия позиции
        if not in_position:
            if close > line_up: #high
                potential = (line_up - line_down) / line_up * 100
                if potential > 0.5:
                    # открываем позицию в шорт
                    enter_short = line_up
                    in_position = True
                    take_short = line_down
                    df = add_new_position('Short', time, line_up, size)
                    mae = mfe = line_up
            elif close < line_down: #low
                potential = (line_up - line_down) / line_down * 100
                if potential > 0.5:
                    enter_long = line_down
                    in_position = True
                    take_long = line_up
                    df = add_new_position('Long', time, line_down, size)
                    mae = mfe = line_down
        else:
            # сначала проверяем на условие стоп лосса
            if sma > enter_short > 0.0:
                # закрываемся
                in_position = False
                enter_short = 0.0
                if close < mfe:
                    mfe = close
                df = close_new_position(df, 'stop', time, close, 'short', mae, mfe)
                result_df = pd.concat([result_df, df], ignore_index=True)
            elif 0.0 < sma < enter_long:
                in_position = False
                enter_long = 0.0
                if close < mfe:
                    mfe = close
                df = close_new_position(df, 'stop', time, close, 'long', mae, mfe)
                result_df = pd.concat([result_df, df], ignore_index=True)

            # если не отстопило, проверяем на условия закрытия по тейку
            else:
                if close < line_down and enter_short > 0.0: #low
                    in_position = False
                    df = close_new_position(df, 'take to line', time, line_down, 'short', mae, mfe)
                    result_df = pd.concat([result_df, df], ignore_index=True)
                    enter_short = take_short = 0.0
                elif close < take_short: #low
                    in_position = False
                    df = close_new_position(df, 'take to limit', time, take_short, 'short', mae, mfe)
                    result_df = pd.concat([result_df, df], ignore_index=True)
                    enter_short = take_short = 0.0
                elif close > line_up and enter_long > 0.0: #high
                    in_position = False
                    enter_long = take_long = 0.0
                    df = close_new_position(df, 'take to line', time, line_up, 'long', mae, mfe)
                    result_df = pd.concat([result_df, df], ignore_index=True)
                elif close > take_long > 0.0: #high
                    in_position = False
                    df = close_new_position(df, 'take to limit', time, take_long, 'long', mae, mfe)
                    result_df = pd.concat([result_df, df], ignore_index=True)
                    enter_long = take_long = 0.0
                elif (close > mae and enter_short > 0.0) | (close < mae and enter_long > 0.0):
                    mae = close
                elif (close < mfe and enter_short > 0.0) | (close > mfe and enter_long > 0.0):
                    mfe = close

    if len(result_df) > 0:
        result_df['cumulat_per'] = round(result_df['result_per'].cumsum(), 2)
        result_df['cum_max_per'] = round(result_df['cumulat_per'].cummax(), 2)
        result_df['drawdown'] = result_df['cumulat_per'] - result_df['cum_max_per']

        result_df.to_csv(r'.\reports\test_result.csv', index=False, sep="\t")
        total = result_df['result_per'].sum()
        drawdown = result_df['drawdown'].min()
        print(f'Drawdown = {drawdown}')
        print(f'Total PnL = {total}')
    return result_df


def strategy_zscore(coin1, coin2, start_date, end_date, lookback, level):

    start_date = start_date - lookback * tf_5m  # для того, что бы расчет стратегии начался с правильных показаний индик.
    # получение исходных данных
    if end_date is None:
        end_date = datetime.datetime.now().timestamp()
    df_coin1 = modul.get_sql_history_price(coin1, connection, start_date, end_date)
    df_coin2 = modul.get_sql_history_price(coin2, connection, start_date, end_date)
    spread_df = modul.make_spread_df(df_coin1, df_coin2, last_to_end=True, tf=tf_5m)

    spread_df = ind.zscore_calculating(spread_df, lookback)
    total, total_per, per_no_commis = 0, 0, 0

    spread_df['zsc_shift'] = spread_df.shift(periods=1)['zscore']
    # находим пересечения 0 и уровней Up/down, в местах пересечения ставим метки открытия/закрытия
    spread_df['going_to'] = np.where(((spread_df['zscore'] > 0) & (spread_df['zsc_shift'] <= 0)) |
                                   ((spread_df['zscore'] < 0) & (spread_df['zsc_shift'] >= 0)),
                                   'zero',
                                   spread_df['zsc_shift'])
    spread_df['going_to'] = np.where(((spread_df['zscore'] > level) & (spread_df['zsc_shift'] <= level)),
                                   'DOWN',
                                   spread_df['going_to'])
    spread_df['going_to'] = np.where(((spread_df['zscore'] < -level) & (spread_df['zsc_shift'] >= -level)),
                                   'UP',
                                   spread_df['going_to'])

    # остальные строки удаляем, как не нужные, сдвигаем дф еще раз
    spread_df = spread_df[
        (spread_df.going_to == 'zero') | (spread_df.going_to == 'UP') | (spread_df.going_to == 'DOWN')]
    spread_df['cross_shift'] = spread_df.shift(periods=1)['going_to']

    # оставляем строки, где сигнал изменился (например с UP на zero)
    spread_df = spread_df[spread_df.cross_shift != spread_df.going_to]

    # высчитываем разницу от открытия до закрытия, сумму и процент
    spread_df['close_shift'] = spread_df.shift(periods=-1)['close']
    spread_df['result'] = np.where(spread_df.going_to == 'UP',
                                 round(spread_df['close_shift'] - spread_df['close'], 6),
                                 spread_df['cross_shift'])
    spread_df['result'] = np.where(spread_df.going_to == 'DOWN',
                                 round(spread_df['close'] - spread_df['close_shift'], 6),
                                 spread_df['result'])

    result_df = spread_df[spread_df.going_to != 'zero']
    result_df['mae_per'] = 0.0
    result_df['result'] = result_df['result'].astype(float)
    result_df['result_per'] = round((result_df['result'] / result_df['close'] * 100), 2)
    result_df['result_per_no'] = result_df['result_per'] - 0.16
    result_df.rename(columns={'startTime': 'open_time'}, inplace=True)
    if len(result_df) > 0:
        result_df['cumulat_per'] = round(result_df['result_per'].cumsum(), 2)
        result_df['cum_max_per'] = round(result_df['cumulat_per'].cummax(), 2)
        result_df['drawdown'] = result_df['cumulat_per'] - result_df['cum_max_per']

        result_df.to_csv(r'.\reports\test_result.csv', index=False, sep="\t")
        total = result_df['result_per'].sum()
        drawdown = result_df['drawdown'].min()
        print(f'Drawdown = {drawdown}')
        print(f'Total PnL = {total}')
    return result_df


def strategy_lregress_channel(coin1, coin2, start_date, end_date, lookback, std_dev):
    start_date = start_date - lookback * tf_5m  # для того, что бы расчет стратегии начался с правильных показаний индик.
    # получение исходных данных
    if end_date is None:
        end_date = datetime.datetime.now().timestamp()
    df_coin1 = modul.get_sql_history_price(coin1, connection, start_date, end_date)
    df_coin2 = modul.get_sql_history_price(coin2, connection, start_date, end_date)
    full_df = modul.make_spread_df(df_coin1, df_coin2, last_to_end=True, tf=tf_5m)

    result_df = df = pd.DataFrame()
    in_position = wait_sma_cross = False
    enter_short = enter_long = 0.0
    mae = mfe = stop = 0.0
    stop_per = 0.1
    amount = 250.0

    full_df = modul.rolling_st_dev_channels(full_df, lookback, std_dev)
    spread_df = full_df.iloc[lookback:]
    # loop = False
    loop = True
    if loop:
        for index in range(len(spread_df)):
            # т.к. данный индикатор меняет значения, надо его пересчитывать внутри цикла
            # df_for_ind = full_df[index:lookback+index]
            # df_channel = modul.standard_deviation_channels(df_for_ind, lookback, std_dev)
            # df_channel = indicators.get_stdev_channels(df_for_ind, lookback, level)
            # line_up = df_channel.iloc[len(df_channel)-1]['line_up']
            # line_down = df_channel.iloc[len(df_channel)-1]['line_down']
            # line_center = df_channel.iloc[len(df_channel)-1]['line_center']
            line_up = spread_df.iloc[index]['line_up']
            line_down = spread_df.iloc[index]['line_down']
            line_center = spread_df.iloc[index]['line_center']

            # вынем из дф нужные данные в переменные
            close = spread_df.iloc[index]['close']
            time = spread_df.iloc[index]['startTime']
            if index > 0:
                close_before = spread_df.iloc[index-1]['close']
            else:
                close_before = spread_df.iloc[index]['close']
            size = amount/close

            if wait_sma_cross:
                if (close_before < line_center < close) | (close_before > line_center > close):
                    wait_sma_cross = False
            elif not wait_sma_cross:
                # сначала смотрим условия для открытия позиции
                if not in_position:
                    if close_before > line_up > close:
                    # if close > line_up:
                        potential = (line_up - line_center) / line_up * 100
                        # if potential > 0.5:
                        # открываем позицию в шорт
                        enter_short = line_up
                        in_position = True
                        stop = line_up + line_up*stop_per
                        df = add_new_position('short', time, close, size)
                        mae = mfe = line_up
                    elif close > line_down > close_before:
                    # elif close < line_down:
                        potential = (line_center - line_down) / line_down * 100
                        # if potential > 0.5:
                        enter_long = line_down
                        in_position = True
                        stop = line_down - line_down*stop_per
                        df = add_new_position('long', time, close, size)
                        mae = mfe = line_down
                else:
                    # сначала проверяем на условие стоп лосса
                    if close > stop and enter_short > 0.0:
                        # закрываемся
                        in_position = False
                        wait_sma_cross = True
                        if close < mfe:
                            mfe = close
                        df = close_new_position(df, 'stop', time, stop, 'short', mae, mfe)
                        result_df = pd.concat([result_df, df], ignore_index=True)
                        enter_short = stop = 0.0
                    elif close < stop and enter_long > 0.0:
                        in_position = False
                        wait_sma_cross = True
                        df = close_new_position(df, 'stop', time, stop, 'long', mae, mfe)
                        result_df = pd.concat([result_df, df], ignore_index=True)
                        enter_long = stop = 0.0
                    else:
                        # Проверим не пора ли закрывать позицию по тейку
                        # if close < line_down and enter_short > 0.0:
                        if close < line_center and enter_short > 0.0:
                            in_position = False
                            enter_short = stop = 0.0
                            df = close_new_position(df, 'take', time, close, 'short', mae, mfe)
                            result_df = pd.concat([result_df, df], ignore_index=True)
                        # elif close > line_up and enter_long > 0.0:
                        elif close > line_center and enter_long > 0.0:
                            in_position = False
                            enter_long = stop = 0.0
                            df = close_new_position(df, 'take', time, close, 'long', mae, mfe)
                            result_df = pd.concat([result_df, df], ignore_index=True)
                        elif (close > mae and enter_short > 0.0) | (close < mae and enter_long > 0.0):
                            mae = close
                        elif (close < mfe and enter_short > 0.0) | (close > mfe and enter_long > 0.0):
                            mfe = close
        if in_position:
            if enter_short > 0.0:
                df = close_new_position(df, 'time', time, close, 'short', mae, mfe)
                result_df = pd.concat([result_df, df], ignore_index=True)
            else:
                df = close_new_position(df, 'time', time, close, 'long', mae, mfe)
                result_df = pd.concat([result_df, df], ignore_index=True)
    else:
    # if loop:
        spread_df.drop(["time", "open", "high", "low", "std_dev"], axis=1, inplace=True)
        spread_df['close_prev'] = spread_df.shift(periods=1)['close']
        # находим пересечения 0 и уровней Up/down, в местах пересечения ставим метки открытия/закрытия
        spread_df['going_to'] = np.where(((spread_df['close'] > spread_df['line_center'])
                                          & (spread_df['close_prev'] <= spread_df['line_center'])) |
                                         ((spread_df['close'] < spread_df['line_center'])
                                          & (spread_df['close_prev'] >= spread_df['line_center'])),
                                         'zero',
                                         spread_df['close_prev'])
        spread_df['going_to'] = np.where((spread_df['close'] > spread_df['line_up']),
                                         'DOWN',
                                         spread_df['close_prev'])
        spread_df['going_to'] = np.where((spread_df['close'] < spread_df['line_down']),
                                         'UP',
                                         spread_df['close_prev'])

        # остальные строки удаляем, как не нужные, сдвигаем дф еще раз
        spread_df = spread_df[
            (spread_df.going_to == 'zero') | (spread_df.going_to == 'UP') | (spread_df.going_to == 'DOWN')]
        spread_df['cross_shift'] = spread_df.shift(periods=1)['going_to']

        # оставляем строки, где сигнал изменился (например с UP на zero)
        spread_df = spread_df[spread_df.cross_shift != spread_df.going_to]

        # высчитываем разницу от открытия до закрытия, сумму и процент
        spread_df['cls_price'] = spread_df.shift(periods=-1)['close']
        spread_df['cls_time'] = spread_df.shift(periods=-1)['startTime']
        spread_df['result'] = np.where(spread_df.going_to == 'UP',
                                       round(spread_df['cls_price'] - spread_df['close'], 6),
                                       spread_df['cross_shift'])
        spread_df['result'] = np.where(spread_df.going_to == 'DOWN',
                                       round(spread_df['close'] - spread_df['cls_price'], 6),
                                       spread_df['result'])

        spread_df = spread_df[spread_df.going_to != 'zero']
        if len(spread_df) == 1:
            if spread_df.iloc[0]['cls_price'] > 0:
                pass
            else:
                spread_df = spread_df.head(0)
        else:
            spread_df['mae_per'] = 0.0
            spread_df['result'] = spread_df['result'].astype(float)
            spread_df['result_per'] = round((spread_df['result'] / spread_df['close'] * 100), 2)
            spread_df['result_per_no'] = spread_df['result_per'] - 0.16
            spread_df.rename(columns={'startTime': 'open_time'}, inplace=True)
        result_df = spread_df.copy()

    if len(result_df) > 0:
        result_df['cumulat_per'] = round(result_df['result_per'].cumsum(), 2)
        result_df['cum_max_per'] = round(result_df['cumulat_per'].cummax(), 2)
        result_df['drawdown'] = result_df['cumulat_per'] - result_df['cum_max_per']

        result_df.to_csv(r'.\reports\test_result.csv', index=False, sep="\t")
        total = result_df['result_per'].sum()
        drawdown = result_df['drawdown'].min()
        print(f'Drawdown = {drawdown}')
        print(f'Total PnL = {total}')

    return result_df


def strategy_structurebreak_catcher(coin1, coin2, start_date, end_date, lookback, sigma):
    """
    Дожидаемся ситуации, когда цена начинает плавно уходить вверх/вниз, и заходим в направлении цены,
    половина - когда идет выход за ББ, вторая половина - если будет касание sma.
    Стоп - на противоположной линии ББ, Тейк - пока не решил))
    :param coin1:
    :param coin2:
    :param start_date:
    :param end_date:
    :param lookback:
    :param sigma:
    :return:
    """

    start_date = start_date - lookback * 2 * tf_5m  # для того, что бы расчет стратегии начался с правильных показаний индик.
    end_date = end_date + lookback * 2 * tf_5m  #в данной стратегии сделка может длиться долго
    df_coin1 = modul.get_sql_history_price(coin1, connection, start_date, end_date)
    df_coin2 = modul.get_sql_history_price(coin2, connection, start_date, end_date)
    spread_df = modul.make_spread_df(df_coin1, df_coin2, last_to_end=True, tf=tf_5m)
    result_df = df = pd.DataFrame()

    spread_df['bb_up'], spread_df['sma'], spread_df['bb_down'] = talib.BBANDS(spread_df.close, lookback, sigma, sigma, 0)
    spread_df['atr'] = talib.ATR(spread_df['high'], spread_df['low'], spread_df['close'], timeperiod=lookback/2)
    in_position = False
    last_short = last_long = 0.0
    mae = mfe = 0.0
    amount = 250.0
    check_df = spread_df.copy()
    # проверим, что в отбор не попали заведомо не пригодные пары
    # _, _, time_to_opposite = modul.check_for_touch_bb(check_df[:lookback * 2], lookback, sigma)
    # if time_to_opposite > lookback / 2:
    #     return result_df
    spread_df = spread_df.iloc[lookback*2:]
    spread_df = spread_df.reset_index()
    high, low = ind.williams_fractals(spread_df, 5)
    for index in range(len(spread_df)):

        # вынем из дф нужные данные в переменные
        if index > 0:
            close_before = spread_df.iloc[index-1]['close']
        else:
            close_before = spread_df.iloc[index]['close']
        close = spread_df.iloc[index]['close']
        time = spread_df.iloc[index]['startTime']
        bb_up = spread_df.iloc[index]['bb_up']
        bb_down = spread_df.iloc[index]['bb_down']
        sma = spread_df.iloc[index]['sma']
        size = amount / close
        take = 0.05
        # сначала смотрим условия для открытия позиции
        if not in_position:
            _, _, time_to_opposite = modul.check_for_touch_bb(check_df[lookback:(lookback*2+index)], lookback, sigma)
            if time_to_opposite < lookback / 2:
               continue
            # Проверяем на условие первого входа
            if close > sma:
                if close > bb_up > close_before:
                    atr_touch = check_df.iloc[(lookback*2+index)-int(time_to_opposite)]['atr']
                    atr = spread_df.iloc[index]['atr']
                    if atr_touch > atr:
                        # potential = (bb_up - sma) / bb_up * 100
                        # if potential > 0.5:
                        # открываем позицию в лонг
                        in_position = True
                        df = add_new_position('bb-1 long', time, close, size/2)
                        mae = mfe = last_long = close
            elif close < sma:
                if close < bb_down < close_before:
                    atr_touch = check_df.iloc[(lookback*2+index)-int(time_to_opposite)]['atr']
                    atr = spread_df.iloc[index]['atr']
                    if atr_touch > atr:
                        # открываем позицию в short
                        in_position = True
                        df = add_new_position('bb-1 short', time, close, size/2)
                        mae = mfe = last_short = close
        else:
            if last_short > 0.0:
                if close > sma > close_before and len(df) == 1:
                    # Если не было докупки на sma, то докупаем
                    if close < mfe:
                        mfe = close
                    new_row2 = add_new_position(f'short sma', time, close, size/2)
                    df = pd.concat([df, new_row2], ignore_index=True)
                elif (last_short - close)/close > take:
                    # Закрываем по тейку
                    in_position = False
                    if close < mfe:
                        mfe = close
                    df = close_new_position(df, 'take', time, close, 'short', mae, mfe)
                    result_df = pd.concat([result_df, df], ignore_index=True)
                    last_short = mae = mfe = 0.0
                elif close > bb_up > close_before:
                    # Закрываем по стопу
                    in_position = False
                    if close < mfe:
                        mfe = close
                    df = close_new_position(df, 'stop', time, close, 'short', mae, mfe)
                    result_df = pd.concat([result_df, df], ignore_index=True)
                    last_short = mae = mfe = 0.0
                elif close > mae:
                    mae = close
                elif close < mfe:
                    mfe = close
            else:
                if close < sma < close_before and len(df) == 1:
                    # Если не было докупки на sma, то докупаем
                    if close > mfe:
                        mfe = close
                    new_row2 = add_new_position(f'long sma', time, close, size / 2)
                    df = pd.concat([df, new_row2], ignore_index=True)
                elif (close - last_long)/close > take:
                    # Закрываем по тейку
                    in_position = False
                    if close > mfe:
                        mfe = close
                    df = close_new_position(df, 'take', time, close, 'long', mae, mfe)
                    result_df = pd.concat([result_df, df], ignore_index=True)
                    last_long = mae = mfe = 0.0
                elif close < bb_down < close_before:
                    # Закрываем по стопу
                    in_position = False
                    if close > mfe:
                        mfe = close
                    df = close_new_position(df, 'stop', time, close, 'long', mae, mfe)
                    result_df = pd.concat([result_df, df], ignore_index=True)
                    last_long = mae = mfe = 0.0
                elif close < mae:
                    mae = close
                elif close > mfe:
                    mfe = close
    if in_position:
        if last_short > 0.0:
            df = close_new_position(df, 'time', time, close, 'short', mae, mfe)
            result_df = pd.concat([result_df, df], ignore_index=True)
        else:
            df = close_new_position(df, 'time', time, close, 'long', mae, mfe)
            result_df = pd.concat([result_df, df], ignore_index=True)

    if len(result_df) > 0:
        result_df['cumulat_per'] = result_df['result_per'].cumsum()
        result_df['cum_max_per'] = result_df['cumulat_per'].cummax()
        result_df['drawdown'] = result_df['cumulat_per'] - result_df['cum_max_per']

        result_df.to_csv(r'.\reports\test_result.csv', index=False, sep="\t")
        total = result_df['result_per'].sum()
        drawdown = result_df['drawdown'].min()
        print(f'Drawdown = {drawdown}')
        print(f'Total PnL = {total}')
    return result_df


def strategy_pp_supertrend(coin1, start_date, end_date, pp_prd, atr_factor, atr_prd):
    # start_date = start_date - lookback * 2 * tf_5m  # для того, что бы расчет стратегии начался с правильных показаний индик.
    # end_date = end_date + lookback * 2 * tf_5m  # в данной стратегии сделка может длиться долго
    spread_df = modul.get_sql_history_price(coin1, connection, start_date, end_date)
    # spread_df = modul.convert_to_tf(spread_df, 900) #15 min timeframe
    if len(spread_df) == 0:
        return pd.DataFrame()

    spread_df.sort_values(by='time', ascending=True, inplace=True, ignore_index=True)
    # df_coin2 = modul.get_sql_history_price(coin2, connection, start_date, end_date)
    # spread_df = modul.make_spread_df(df_coin1, df_coin2, last_to_end=True, tf=tf_5m)
    result_df = df = pd.DataFrame()

    spread_df = ind.pivot_point_supertrend(spread_df, pp_prd, atr_factor, atr_prd)
    in_position = False
    last_short = last_long = 0.0
    mae = mfe = 0.0
    amount = 250.0
    check_df = spread_df.copy()

    spread_df = spread_df.reset_index()
    stop = 0.0
    for index in range(len(spread_df)):

        # вынем из дф нужные данные в переменные
        close = spread_df.iloc[index]['close']
        time = spread_df.iloc[index]['startTime']
        trend = spread_df.iloc[index]['trend']
        size = amount / close
        # сначала смотрим условия для открытия позиции
        if not in_position:
            if spread_df.iloc[index]['switch']:
                # Проверяем на условие первого входа

                if spread_df.iloc[index]['switch_to'] == 'up':
                    # переключились на растущий тренд, смотрим два предыдущих
                    test_df = check_df[:index]
                    test_df = test_df[test_df['switch_to'] == 'up']

                    if len(test_df) > 1:
                        if trend > test_df.iloc[-1]['trend'] > test_df.iloc[-2]['trend']:
                            #текущий тренд - третий подряд растущий, открываем позицию лонг
                            in_position = True
                            df = add_new_position('long', time, close, size)
                            stop = trend
                            mae = mfe = last_long = close
                else:
                    # переключились на падающий тренд, смотрим два предыдущих
                    test_df = check_df[:index]
                    test_df = test_df[test_df['switch_to'] == 'down']

                    if len(test_df) > 1:
                        if trend < test_df.iloc[-1]['trend'] < test_df.iloc[-2]['trend']:
                            # текущий тренд - третий подряд падающий, открываем позицию шорт
                            in_position = True
                            df = add_new_position('short', time, close, size)
                            stop = trend
                            mae = mfe = last_short = close
        else:
            if last_short > 0:
                # сначала проверяем, не отстопило ли
                if close > stop:
                    in_position = False
                    if close < mfe:
                        mfe = close
                    df = close_new_position(df, 'stop', time, stop, 'short', mae, mfe)
                    result_df = pd.concat([result_df, df], ignore_index=True)
                    last_short = mae = mfe = stop = 0.0
                    if close > mae:
                        mae = close
                    elif close < mfe:
                        mfe = close
                else:
                    # тогда смотрим не пора ли передвинуть стоп
                    test_df = check_df[:index]
                    test_df = test_df[test_df['switch_to'] == 'down']
                    if test_df.iloc[-1]['trend'] < stop:
                        stop = test_df.iloc[-1]['trend']
            else:
                # сначала проверяем, не отстопило ли
                if close < stop:
                    in_position = False
                    if close < mfe:
                        mfe = close
                    df = close_new_position(df, 'stop', time, stop, 'long', mae, mfe)
                    last_long = mae = mfe = stop = 0.0
                    result_df = pd.concat([result_df, df], ignore_index=True)
                    if close > mae:
                        mae = close
                    elif close < mfe:
                        mfe = close
                else:
                    # тогда смотрим не пора ли передвинуть стоп
                    test_df = check_df[:index]
                    test_df = test_df[test_df['switch_to'] == 'up']
                    if test_df.iloc[-1]['trend'] > stop:
                        stop = test_df.iloc[-1]['trend']

                    if close > mae:
                        mae = close
                    elif close < mfe:
                        mfe = close

    if in_position:
        if last_short > 0.0:
            df = close_new_position(df, 'time', time, close, 'short', mae, mfe)
            result_df = pd.concat([result_df, df], ignore_index=True)
        else:
            df = close_new_position(df, 'time', time, close, 'long', mae, mfe)
            result_df = pd.concat([result_df, df], ignore_index=True)

    if len(result_df) > 0:
        result_df['cumulat_per'] = result_df['result_per'].cumsum()
        result_df['cum_max_per'] = result_df['cumulat_per'].cummax()
        result_df['drawdown'] = result_df['cumulat_per'] - result_df['cum_max_per']

        result_df.to_csv(r'.\reports\test_result.csv', index=False, sep="\t")
        total = result_df['result_per'].sum()
        drawdown = result_df['drawdown'].min()
        print(f'Drawdown = {drawdown}')
        print(f'Total PnL = {total}')

    return result_df


def strategy_pp_supertrend_v2(coin1, start_date, end_date, pp_prd, atr_factor, atr_prd):

    start_date = start_date - 500 * tf_5m  # для того, что бы расчет стратегии начался с правильных показаний индик.
    spread_df = modul.get_sql_history_price(coin1, connection, start_date, end_date)
    spread_df = modul.convert_to_tf(spread_df, 900) #15 min timeframe
    if len(spread_df) == 0:
        return pd.DataFrame()

    spread_df.sort_values(by='time', ascending=True, inplace=True, ignore_index=True)
    result_df = df = pd.DataFrame()

    spread_df = ind.pivot_point_supertrend(spread_df, pp_prd, atr_factor, atr_prd)
    in_position = False
    last_short = last_long = 0.0
    mae = mfe = 0.0
    amount = 250.0

    spread_df = spread_df.iloc[500:]
    spread_df = spread_df.reset_index()
    check_df = spread_df.copy()
    stop = 0.0
    take = 0.0
    for index in range(len(spread_df)):

        # вынем из дф нужные данные в переменные
        close = spread_df.iloc[index]['close']
        time = spread_df.iloc[index]['startTime']
        trend = spread_df.iloc[index]['trend']
        size = amount / close
        # сначала смотрим условия для открытия позиции
        if not in_position:
            if spread_df.iloc[index]['switch']:
                # Проверяем на условие первого входа
                if spread_df.iloc[index]['switch_to'] == 'down':
                    # переключились на растущий тренд, смотрим два предыдущих
                    test_df = check_df[:index]
                    test_df = test_df[test_df['switch_to'] == 'up']

                    if len(test_df) > 1:
                        if test_df.iloc[-1]['trend'] > test_df.iloc[-2]['trend']:  # v.2
                            difference = (spread_df.iloc[index - 1]['trend'] - test_df.iloc[-1]['trend']) / test_df.iloc[-1][
                                'trend']
                            if difference > 0.005:
                                # текущий тренд - третий подряд растущий, открываем позицию лонг
                                in_position = True
                                df = add_new_position('long', time, close, size)
                                stop = test_df.iloc[-1]['trend']
                                mae = mfe = last_long = close
                else:
                    # переключились на падающий тренд, смотрим два предыдущих
                    test_df = check_df[:index]
                    test_df = test_df[test_df['switch_to'] == 'down']

                    if len(test_df) > 1:
                        if test_df.iloc[-1]['trend'] < test_df.iloc[-2]['trend']:
                            difference = (test_df.iloc[-1]['trend'] - spread_df.iloc[index - 1]['trend']) / \
                                         spread_df.iloc[index - 1]['trend']
                            if difference > 0.005:
                                # текущий тренд - третий подряд падающий, открываем позицию шорт
                                in_position = True
                                df = add_new_position('short', time, close, size)
                                stop = test_df.iloc[-1]['trend']  # v.2
                                mae = mfe = last_short = close
        else:
            if last_short > 0:
                # сначала проверяем, не отстопило ли
                if close > stop:
                    in_position = False
                    if close < mfe:
                        mfe = close
                    df = close_new_position(df, 'stop', time, stop, 'short', mae, mfe)
                    result_df = pd.concat([result_df, df], ignore_index=True)
                    last_short = mae = mfe = stop = 0.0
                    if close > mae:
                        mae = close
                    elif close < mfe:
                        mfe = close
                else:
                    # тогда смотрим не пора ли передвинуть стоп
                    test_df = check_df[:index]
                    test_df = test_df[test_df['switch_to'] == 'down']
                    if test_df.iloc[-1]['trend'] < stop:
                        stop = test_df.iloc[-1]['trend']
            else:
                # сначала проверяем, не отстопило ли
                if close < stop:
                    in_position = False
                    if close < mfe:
                        mfe = close
                    df = close_new_position(df, 'stop', time, stop, 'long', mae, mfe)
                    last_long = mae = mfe = stop = 0.0
                    result_df = pd.concat([result_df, df], ignore_index=True)
                    if close > mae:
                        mae = close
                    elif close < mfe:
                        mfe = close
                else:
                    # тогда смотрим не пора ли передвинуть стоп
                    test_df = check_df[:index]
                    test_df = test_df[test_df['switch_to'] == 'up']
                    if test_df.iloc[-1]['trend'] > stop:
                        stop = test_df.iloc[-1]['trend']

                    if close > mae:
                        mae = close
                    elif close < mfe:
                        mfe = close

    if in_position:
        if last_short > 0.0:
            df = close_new_position(df, 'time', time, close, 'short', mae, mfe)
            result_df = pd.concat([result_df, df], ignore_index=True)
        else:
            df = close_new_position(df, 'time', time, close, 'long', mae, mfe)
            result_df = pd.concat([result_df, df], ignore_index=True)

    if len(result_df) > 0:
        result_df['cumulat_per'] = result_df['result_per'].cumsum()
        result_df['cum_max_per'] = result_df['cumulat_per'].cummax()
        result_df['drawdown'] = result_df['cumulat_per'] - result_df['cum_max_per']

        result_df.to_csv(r'.\reports\test_result.csv', index=False, sep="\t")
        total = result_df['result_per'].sum()
        drawdown = result_df['drawdown'].min()
        print(f'Drawdown = {drawdown}')
        print(f'Total PnL = {total}')

    return result_df


def flat_filter(df, direction):
    """
    На входе df с предрасчитынным pp_supertrend
    Считаем что флет, если за последние 8 смен рр-тренда ни одно начало тренда UP не было
    выше чем начало тренда DOWN, и наоборот.
    :return: bool
    """
    # Так как 80% рынок во флете, то по умолчанию считаем что сейчас флет, если не доказано обратное.
    flat = True
    if len(df) > 0:
        df['prev_up'] = df['trend_up'].shift(1)
        df['prev_down'] = df['trend_down'].shift(1)
        df = df[df['switch'] == True]
        # Проверяем, есть ли вынос на последнем тренде
        if direction == 'up':
            if df.iloc[-1]['prev_down'] < df.iloc[-3]['trend']:
                flat = False
        else:
            if df.iloc[-1]['prev_up'] > df.iloc[-3]['trend']:
                flat = False
        # Если выноса нет, смотрим последние 4 пары UP/DOWN
        if flat:
            df_up = df[df['switch_to'] == 'up'].tail(3)
            df_down = df[df['switch_to'] == 'down'].tail(3)
            max_up = df_up['trend'].max()
            min_down = df_down['trend'].min()
            if max_up > min_down:
                flat = False

    return flat


def strategy_pp_supertrend_v4(coin1, start_date, end_date, pp_prd, atr_factor, atr_prd):

    start_date = start_date - 500 * tf_5m  # для того, что бы расчет стратегии начался с правильных показаний индик.
    spread_df = modul.get_sql_history_price(coin1, connection, start_date, end_date)
    # spread_df = modul.convert_to_tf(spread_df, 900) #15 min timeframe
    if len(spread_df) == 0:
        return pd.DataFrame()

    spread_df.sort_values(by='time', ascending=True, inplace=True, ignore_index=True)
    result_df = df = pd.DataFrame()

    spread_df = ind.pivot_point_supertrend(spread_df, pp_prd, atr_factor, atr_prd)
    in_position = False
    last_short = last_long = 0.0
    mae = mfe = 0.0
    amount = 250.0

    # spread_df = spread_df.iloc[500:]
    # spread_df = spread_df.reset_index()
    check_df = spread_df.copy()
    stop = 0.0
    # take = 0.0
    # is_it_flat = False
    for index in range(500,len(spread_df)):

        # вынем из дф нужные данные в переменные
        close = spread_df.iloc[index]['close']
        time = spread_df.iloc[index]['startTime']
        trend = spread_df.iloc[index]['trend']
        size = amount / close
        # сначала смотрим условия для открытия позиции
        if not in_position:
            if spread_df.iloc[index]['switch']:
                # Проверяем на условие первого входа
                if spread_df.iloc[index]['switch_to'] == 'down':
                    # переключились на растущий тренд, смотрим два предыдущих
                    test_df = check_df[:index]
                    test_df = test_df[test_df['switch_to'] == 'up']

                    if len(test_df) > 1:
                        if test_df.iloc[-1]['trend'] > test_df.iloc[-2]['trend']:  # v.2
                            is_it_flat = flat_filter(spread_df[:index+1], 'down')
                            if not is_it_flat:
                                difference = (spread_df.iloc[index - 1]['trend'] - test_df.iloc[-1]['trend']) / test_df.iloc[-1][
                                    'trend']
                                if difference > 0.005:
                                    in_position = True
                                    df = add_new_position('long', time, close, size)
                                    stop = test_df.iloc[-1]['trend']
                                    mae = mfe = last_long = close
                else:
                    # переключились на падающий тренд, смотрим два предыдущих
                    test_df = check_df[:index]
                    test_df = test_df[test_df['switch_to'] == 'down']
                    if len(test_df) > 1:
                        if test_df.iloc[-1]['trend'] < test_df.iloc[-2]['trend']:
                            is_it_flat = flat_filter(spread_df[:index+1], 'up')
                            if not is_it_flat:
                                difference = (test_df.iloc[-1]['trend'] - spread_df.iloc[index - 1]['trend']) / \
                                             spread_df.iloc[index - 1]['trend']
                                if difference > 0.005:
                                    in_position = True
                                    df = add_new_position('short', time, close, size)
                                    stop = test_df.iloc[-1]['trend']  # v.2
                                    mae = mfe = last_short = close
        else:
            if last_short > 0:
                # сначала проверяем, не отстопило ли
                if close > stop:
                    in_position = False
                    if close < mfe:
                        mfe = close
                    df = close_new_position(df, 'stop', time, stop, 'short', mae, mfe)
                    result_df = pd.concat([result_df, df], ignore_index=True)
                    last_short = mae = mfe = stop = 0.0
                    if close > mae:
                        mae = close
                    elif close < mfe:
                        mfe = close
                else:
                    # тогда смотрим не пора ли передвинуть стоп
                    test_df = check_df[:index]
                    test_df = test_df[test_df['switch_to'] == 'down']
                    if test_df.iloc[-1]['trend'] < stop:
                        stop = test_df.iloc[-1]['trend']
            else:
                # сначала проверяем, не отстопило ли
                if close < stop:
                    in_position = False
                    if close < mfe:
                        mfe = close
                    df = close_new_position(df, 'stop', time, stop, 'long', mae, mfe)
                    last_long = mae = mfe = stop = 0.0
                    result_df = pd.concat([result_df, df], ignore_index=True)
                    if close > mae:
                        mae = close
                    elif close < mfe:
                        mfe = close
                else:
                    # тогда смотрим не пора ли передвинуть стоп
                    test_df = check_df[:index]
                    test_df = test_df[test_df['switch_to'] == 'up']
                    if test_df.iloc[-1]['trend'] > stop:
                        stop = test_df.iloc[-1]['trend']

                    if close > mae:
                        mae = close
                    elif close < mfe:
                        mfe = close

    if in_position:
        if last_short > 0.0:
            df = close_new_position(df, 'time', time, close, 'short', mae, mfe)
            result_df = pd.concat([result_df, df], ignore_index=True)
        else:
            df = close_new_position(df, 'time', time, close, 'long', mae, mfe)
            result_df = pd.concat([result_df, df], ignore_index=True)

    if len(result_df) > 0:
        result_df['cumulat_per'] = result_df['result_per'].cumsum()
        result_df['cum_max_per'] = result_df['cumulat_per'].cummax()
        result_df['drawdown'] = result_df['cumulat_per'] - result_df['cum_max_per']

        result_df.to_csv(r'.\reports\test_result.csv', index=False, sep="\t")
        total = result_df['result_per'].sum()
        drawdown = result_df['drawdown'].min()
        print(f'Drawdown = {drawdown}')
        print(f'Total PnL = {total}')

    return result_df


def test_strategy_pp_supertrend(coin1, start_date, end_date, pp_prd, atr_factor, atr_prd):
    """
    Тестируем стратегию.
    v.1 Не заходить, если расстояние между текущим трендом и предыдущим меньше 0,3%
    Результат - за июль убыток снизился до 700% (был 750%). При >0,5%  результат = -550%
    При > 0,7% -488%, при 1% -477%. То есть макс эффект идет при 0,5%

    v.2 Вход не на начале 3го тренда, а на конце второго.
    Совместно с v.1(>0,5%) результат улучшился до -256%
    Протестировал на 4х месяцах, результат стабильно лучше оригинальной на 30-50%


    """
    start_date = start_date - 500 * tf_5m  # для того, что бы расчет стратегии начался с правильных показаний индик.
    spread_df = modul.get_sql_history_price(coin1, connection, start_date, end_date)
    # spread_df = modul.convert_to_tf(spread_df, 900) #15 min timeframe
    if len(spread_df) == 0:
        return pd.DataFrame()

    spread_df.sort_values(by='time', ascending=True, inplace=True, ignore_index=True)
    # df_coin2 = modul.get_sql_history_price(coin2, connection, start_date, end_date)
    # spread_df = modul.make_spread_df(df_coin1, df_coin2, last_to_end=True, tf=tf_5m)
    result_df = df = pd.DataFrame()

    spread_df = ind.pivot_point_supertrend(spread_df, pp_prd, atr_factor, atr_prd)
    spread_df['bb_up'], _, spread_df['bb_down'] = talib.BBANDS(spread_df.close, 288, 3, 3, 0)
    in_position = False
    last_short = last_long = 0.0
    mae = mfe = 0.0
    amount = 250.0

    spread_df = spread_df.iloc[500:]
    spread_df = spread_df.reset_index()
    check_df = spread_df.copy()
    stop = 0.0
    # take = 0.0
    for index in range(len(spread_df)):

        # вынем из дф нужные данные в переменные
        close = spread_df.iloc[index]['close']
        time = spread_df.iloc[index]['startTime']
        trend = spread_df.iloc[index]['trend']
        size = amount / close
        # сначала смотрим условия для открытия позиции
        if not in_position:
            if spread_df.iloc[index]['switch']:
                # Проверяем на условие первого входа

                # if spread_df.iloc[index]['switch_to'] == 'up':
                if spread_df.iloc[index]['switch_to'] == 'down':  # v.2
                    # переключились на растущий тренд, смотрим два предыдущих
                    test_df = check_df[:index]
                    test_df = test_df[test_df['switch_to'] == 'up']

                    if len(test_df) > 1:
                        if test_df.iloc[-1]['trend'] > test_df.iloc[-2]['trend']:  # v.2
                            is_it_flat = flat_filter(spread_df[:index + 1], 'down')  # v.4
                            if not is_it_flat:
                                difference = (spread_df.iloc[index - 1]['trend'] - test_df.iloc[-1]['trend']) / test_df.iloc[-1]['trend']
                                if difference > 0.005:  # v.1 test
                                    #текущий тренд - третий подряд растущий, открываем позицию лонг
                                    in_position = True
                                    df = add_new_position('long', time, close, size)
                                    stop = test_df.iloc[-1]['trend']  # v.2
                                    mae = mfe = last_long = close
                else:
                    # переключились на падающий тренд, смотрим два предыдущих
                    test_df = check_df[:index]
                    test_df = test_df[test_df['switch_to'] == 'down']

                    if len(test_df) > 1:
                        if test_df.iloc[-1]['trend'] < test_df.iloc[-2]['trend']:  # v.2
                            is_it_flat = flat_filter(spread_df[:index + 1], 'up')  # v.4
                            if not is_it_flat:
                                difference = (test_df.iloc[-1]['trend'] - spread_df.iloc[index-1]['trend']) / spread_df.iloc[index-1]['trend']
                                if difference > 0.005:  # v.1 test
                                    # текущий тренд - третий подряд падающий, открываем позицию шорт
                                    in_position = True
                                    df = add_new_position('short', time, close, size)
                                    stop = test_df.iloc[-1]['trend']  # v.2
                                    mae = mfe = last_short = close
        else:
            if last_short > 0:
                if close > mae:
                    mae = close
                elif close < mfe:
                    mfe = close
                if close > stop:  # сначала проверяем, не отстопило ли
                    in_position = False
                    df = close_new_position(df, 'stop', time, stop, 'short', mae, mfe)
                    result_df = pd.concat([result_df, df], ignore_index=True)
                    last_short = mae = mfe = stop = 0.0
                elif close < spread_df.iloc[index]['bb_down']:  # v.3 теперь проверяем есть ли условие закрытия.
                    in_position = False
                    df = close_new_position(df, 'take', time, close, 'short', mae, mfe)
                    result_df = pd.concat([result_df, df], ignore_index=True)
                    last_short = mae = mfe = stop = 0.0
                else:
                    # тогда смотрим не пора ли передвинуть стоп
                    test_df = check_df[:index]
                    test_df = test_df[test_df['switch_to'] == 'down']
                    if test_df.iloc[-1]['trend'] < stop:
                        stop = test_df.iloc[-1]['trend']
            else:
                if close < mae:
                    mae = close
                elif close > mfe:
                    mfe = close
                if close < stop:  # сначала проверяем, не отстопило ли
                    in_position = False
                    df = close_new_position(df, 'stop', time, stop, 'long', mae, mfe)
                    last_long = mae = mfe = stop = 0.0
                    result_df = pd.concat([result_df, df], ignore_index=True)
                elif close > spread_df.iloc[index]['bb_up']:  # v.3 теперь проверяем есть ли условие закрытия.
                    in_position = False
                    df = close_new_position(df, 'take', time, close, 'long', mae, mfe)
                    result_df = pd.concat([result_df, df], ignore_index=True)
                    last_long = mae = mfe = stop = 0.0
                else:
                    # тогда смотрим не пора ли передвинуть стоп
                    test_df = check_df[:index]
                    test_df = test_df[test_df['switch_to'] == 'up']
                    if test_df.iloc[-1]['trend'] > stop:
                        stop = test_df.iloc[-1]['trend']

    if in_position:
        if last_short > 0.0:
            df = close_new_position(df, 'time', time, close, 'short', mae, mfe)
            result_df = pd.concat([result_df, df], ignore_index=True)
        else:
            df = close_new_position(df, 'time', time, close, 'long', mae, mfe)
            result_df = pd.concat([result_df, df], ignore_index=True)

    if len(result_df) > 0:
        result_df['cumulat_per'] = result_df['result_per'].cumsum()
        result_df['cum_max_per'] = result_df['cumulat_per'].cummax()
        result_df['drawdown'] = result_df['cumulat_per'] - result_df['cum_max_per']

        result_df.to_csv(r'.\reports\test_result.csv', index=False, sep="\t")
        total = result_df['result_per'].sum()
        drawdown = result_df['drawdown'].min()
        print(f'Drawdown = {drawdown}')
        print(f'Total PnL = {total}')

    return result_df


def test_strategy_moex_pp_supertrend(coin1, start_date, end_date, alor_connection, headers, pp_prd, atr_factor, atr_prd):
    """
    Тестируем стратегию v3
    """
    temp_start_date = start_date - 1500 * tf_5m  # для того, что бы расчет стратегии начался с правильных показаний индик.
    spread_df = alor_modul.get_sql_history_price(coin1, alor_connection, temp_start_date, end_date, headers)
    if len(spread_df) == 0:
        return pd.DataFrame()
    spread_df = modul.convert_to_tf(spread_df, 900) #15 min timeframe
    spread_df.sort_values(by='time', ascending=True, inplace=True, ignore_index=True)
    result_df = df = pd.DataFrame()

    spread_df = ind.pivot_point_supertrend(spread_df, pp_prd, atr_factor, atr_prd)
    # spread_df2 = pivot_point_supertrend(spread_df, pp_prd, atr_factor, atr_prd)
    spread_df['bb_up'], _, spread_df['bb_down'] = talib.BBANDS(spread_df.close, 288, 3, 3, 0)
    in_position = False
    last_short = last_long = 0.0
    mae = mfe = 0.0
    amount = 250.0

    start_index = spread_df.index[spread_df['time'] < int(start_date)].tolist()[-1]
    spread_df = spread_df.iloc[start_index:]
    spread_df = spread_df.reset_index()
    check_df = spread_df.copy()
    stop = 0.0
    for index in range(len(spread_df)):

        # вынем из дф нужные данные в переменные
        close = spread_df.iloc[index]['close']
        time = spread_df.iloc[index]['startTime']
        trend = spread_df.iloc[index]['trend']
        size = amount / close
        # сначала смотрим условия для открытия позиции
        if not in_position:
            if spread_df.iloc[index]['switch']:
                # Проверяем на условие первого входа

                # if spread_df.iloc[index]['switch_to'] == 'up':
                if spread_df.iloc[index]['switch_to'] == 'down':  # v.2
                    # переключились на растущий тренд, смотрим два предыдущих
                    test_df = check_df[:index]
                    test_df = test_df[test_df['switch_to'] == 'up']

                    if len(test_df) > 1:
                        if test_df.iloc[-1]['trend'] > test_df.iloc[-2]['trend']:  # v.2
                            difference = (spread_df.iloc[index-1]['trend'] - test_df.iloc[-1]['trend'])/test_df.iloc[-1]['trend']
                            if difference > 0.003:  # v.1 test
                                #текущий тренд - третий подряд растущий, открываем позицию лонг
                                in_position = True
                                df = add_new_position('long', time, close, size)
                                stop = test_df.iloc[-1]['trend']  # v.2
                                mae = mfe = last_long = close
                else:
                    # переключились на падающий тренд, смотрим два предыдущих
                    test_df = check_df[:index]
                    test_df = test_df[test_df['switch_to'] == 'down']

                    if len(test_df) > 1:
                        if test_df.iloc[-1]['trend'] < test_df.iloc[-2]['trend']:  # v.2
                            difference = (test_df.iloc[-1]['trend'] - spread_df.iloc[index-1]['trend']) / spread_df.iloc[index-1]['trend']
                            if difference > 0.003:  # v.1 test
                                # текущий тренд - третий подряд падающий, открываем позицию шорт
                                in_position = True
                                df = add_new_position('short', time, close, size)
                                stop = test_df.iloc[-1]['trend']  # v.2
                                mae = mfe = last_short = close
        else:
            if last_short > 0:
                if close > mae:
                    mae = close
                elif close < mfe:
                    mfe = close
                # сначала проверяем, не отстопило ли
                if close > stop:  # ) or (close < take)
                    in_position = False
                    df = close_new_position(df, 'stop', time, stop, 'short', mae, mfe)
                    result_df = pd.concat([result_df, df], ignore_index=True)
                    last_short = mae = mfe = stop = 0.0
                elif close < spread_df.iloc[index]['bb_down']:  # теперь проверяем есть ли условие закрытия.
                    in_position = False
                    df = close_new_position(df, 'take', time, close, 'short', mae, mfe)
                    result_df = pd.concat([result_df, df], ignore_index=True)
                    last_short = mae = mfe = stop = 0.0
                else:
                    # тогда смотрим не пора ли передвинуть стоп
                    test_df = check_df[:index]
                    test_df = test_df[test_df['switch_to'] == 'down']
                    if test_df.iloc[-1]['trend'] < stop:
                        stop = test_df.iloc[-1]['trend']
            else:
                if close < mae:
                    mae = close
                elif close > mfe:
                    mfe = close
                # сначала проверяем, не отстопило ли
                if close < stop:  # ) or (close > take)
                    in_position = False
                    df = close_new_position(df, 'stop', time, stop, 'long', mae, mfe)
                    last_long = mae = mfe = stop = 0.0
                    result_df = pd.concat([result_df, df], ignore_index=True)
                elif close > spread_df.iloc[index]['bb_up']:  # теперь проверяем есть ли условие закрытия.
                    in_position = False
                    df = close_new_position(df, 'take', time, close, 'long', mae, mfe)
                    result_df = pd.concat([result_df, df], ignore_index=True)
                    last_long = mae = mfe = stop = 0.0
                else:
                    # тогда смотрим не пора ли передвинуть стоп
                    test_df = check_df[:index]
                    test_df = test_df[test_df['switch_to'] == 'up']
                    if test_df.iloc[-1]['trend'] > stop:
                        stop = test_df.iloc[-1]['trend']

    if in_position:
        if last_short > 0.0:
            df = close_new_position(df, 'time', time, close, 'short', mae, mfe)
            result_df = pd.concat([result_df, df], ignore_index=True)
        else:
            df = close_new_position(df, 'time', time, close, 'long', mae, mfe)
            result_df = pd.concat([result_df, df], ignore_index=True)

    if len(result_df) > 0:
        result_df['cumulat_per'] = result_df['result_per'].cumsum()
        result_df['cum_max_per'] = result_df['cumulat_per'].cummax()
        result_df['drawdown'] = result_df['cumulat_per'] - result_df['cum_max_per']

        result_df.to_csv(r'.\reports\test_result.csv', index=False, sep="\t")
        total = result_df['result_per'].sum()
        drawdown = result_df['drawdown'].min()
        print(f'Drawdown = {drawdown}')
        print(f'Total PnL = {total}')

    return result_df


if __name__ == '__main__':
    # start_time = datetime.datetime.now().timestamp() - 2000 * tf_5m
    start_time = datetime.datetime(2023, 3, 15, 0, 0, 0).timestamp()
    end_time = datetime.datetime(2023, 6, 15, 0, 0, 0).timestamp()
    # test_oc_strategy('AAVEUSDT', 'AXSUSDT', start_time, True)
    # test_oc_str_2takes('1000XECUSDT', 'SPELLUSDT', start_time, False)
    # strategy_bb1_3_stop4('1000XECUSDT', 'TRBUSDT', start_time, 1000, end_time)
    # strategy_bb3atr_stop4('GALAUSDT', 'GMTUSDT', start_time, end_time, 240)
    # strategy_pump_catcher('AAVEUSDT', 'AXSUSDT', start_time, end_time, 1000, 0.1, 3)
    # strategy_grid_bb('EGLDUSDT', 'HOTUSDT', start_time, end_time, 1000, 1, 5)
    # strategy_grid_bb_test('EGLDUSDT', 'HOTUSDT', start_time, end_time, 1000, 1, 5)
    # strategy_dev1('ARBUSDT', 'FETUSDT', start_time, end_time, 500, 0.025)
    # strategy_structurebreak_catcher('ARBUSDT', 'FETUSDT', start_time, end_time, 1000, 1)
    # strategy_pp_supertrend_v3('1000XECUSDT', start_time, end_time, 2, 3, 10)
    # check_list_for_strategies(start_time, end_time, 5, 240)

    start_time = datetime.datetime(2023, 3, 15, 0, 0, 0).timestamp()
    end_time = datetime.datetime(2023, 6, 15, 0, 0, 0).timestamp()
    # walk_forward_scaning(start_time, end_time, 9000, 3, 'only_coint')
    # walk_forward_testing(start_time, end_time, 9000, 3, 1000, 'only_coint')
    # walk_forward_testing(start_time, end_time, 2000, 2, 1000, 'only_coint')
    # single_strategy_testing(start_time, end_time)
    single_strategy_testing_moex(start_time, end_time)
    # TODO
    # 1.3 Вход на развороте цены (поискать методы как ловить разворот)
    # 4. Найти обратную стратегию - если цена уходит от средней - заходить по тренду.
    # 6. Доработать АТР стратегию, сделать докупку
