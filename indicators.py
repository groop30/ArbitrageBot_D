import pandas as pd
import numpy as np
import talib
import datetime


def williams_fractals(df, n):
    high_prices = df['high']
    low_prices = df['low']
    ddf_high = pd.DataFrame()
    ddf_high['high'] = high_prices
    ddf_low = pd.DataFrame()
    ddf_low['low'] = low_prices
    mask_list = []
    for num_col in range(1, n+1):
        ddf_high[f'left_{num_col}'] = high_prices.shift(num_col)
        ddf_high[f'right_{num_col}'] = high_prices.shift(-num_col)
        select_1 = (ddf_high[f'left_{num_col}'] <= ddf_high['high'])
        select_2 = (ddf_high[f'right_{num_col}'] < ddf_high['high'])
        ddf_high[f'test_{num_col}'] = select_1 & select_2

        ddf_low[f'left_{num_col}'] = low_prices.shift(num_col)
        ddf_low[f'right_{num_col}'] = low_prices.shift(-num_col)
        select_11 = (ddf_low[f'left_{num_col}'] >= ddf_low['low'])
        select_22 = (ddf_low[f'right_{num_col}'] > ddf_low['low'])
        ddf_low[f'test_{num_col}'] = select_11 & select_22
        mask_list.append(f'test_{num_col}')

    # mask = df.columns.isin(mask_list)
    ddf_high['pivot_high'] = ddf_high.loc[:, mask_list].all(axis=1)
    ddf_low['pivot_low'] = ddf_low.loc[:, mask_list].all(axis=1)

    # return up_fractals, down_fractals
    return ddf_high['pivot_high'], ddf_low['pivot_low']


def pivot_point_supertrend(spread_df, pp_prd, atr_factor, atr_prd):

    # Get high and low prices
    high = spread_df['high']
    low = spread_df['low']
    close = spread_df['close']

    # Calculate pivots
    # pivot_high, pivot_low = williams_fractals(spread_df, pp_prd)
    pivot_high_series, pivot_low_series = williams_fractals(spread_df, pp_prd)
    # Calculate the center line using pivot points

    pivot_high_values = np.where(pivot_high_series, spread_df['high'], False)
    pivot_low_values = np.where(pivot_low_series, spread_df['low'], False)
    all_pivots = np.where(pivot_high_values, pivot_high_values, pivot_low_values)
    pd.Series(all_pivots).replace(0, np.nan, inplace=True)

    spread_df['pivots'] = all_pivots
    list_res = []
    for k in range(pp_prd+1):
        list_res.append(np.nan)

    for i in range((pp_prd+1), len(all_pivots)):
        pivot_now = all_pivots[i - pp_prd]
        if not pd.isna(pivot_now):
            if not pd.isna(list_res[-1]):
                list_res.append((list_res[-1] * 2 + pivot_now) / 3)
            else:
                list_res.append(pivot_now)
        else:
            if not pd.isna(list_res[-1]):
                list_res.append(list_res[-1])
            else:
                list_res.append(pivot_now)

    spread_df['center'] = pd.DataFrame(list_res)

    # Upper/lower bands calculation
    atr = talib.ATR(high, low, close, timeperiod=atr_prd)
    up = spread_df['center'] - (atr_factor * atr)
    dn = spread_df['center'] + (atr_factor * atr)

    l_trend = []
    l_trend_up = []
    l_trend_down = []
    l_switch = []
    l_switch_to = []
    for k in range(atr_prd):
        l_trend.append(np.nan)
        l_trend_up.append(np.nan)
        l_trend_down.append(np.nan)
        l_switch.append(False)
        l_switch_to.append('')

    # задаем точку старта.
    if spread_df.iloc[atr_prd]['close'] < up[atr_prd]:
        l_trend.append(dn[atr_prd])
        l_trend_up.append(up[atr_prd])
        l_trend_down.append(dn[atr_prd])
        l_switch_to.append('down')
        trend_now = 'down'
    else:
        l_trend.append(up[atr_prd])
        l_trend_up.append(up[atr_prd])
        l_trend_down.append(dn[atr_prd])
        l_switch_to.append('up')
        trend_now = 'up'
    l_switch.append(True)

    for i in range(atr_prd+1, len(spread_df)):
        prev_trend = l_trend[-1]
        prev_trend_down = l_trend_down[-1]
        prev_trend_up = l_trend_up[-1]
        close = spread_df.iloc[i]['close']

        # смотрим предыдущие значения, что бы понять, какой был тренд
        if trend_now == 'down':
            # значит был тренд вниз (down)
            if prev_trend >= close:
                l_trend.append(min(prev_trend, dn[i]))
                l_trend_down.append(min(prev_trend, dn[i]))
                l_switch_to.append('')
                l_switch.append(False)
                if prev_trend_up > close:
                    l_trend_up.append(up[i])
                else:
                    l_trend_up.append(max(prev_trend_up, up[i]))
            else:
                # тренд пробит, меняем линию на trend_up
                l_trend.append(max(prev_trend_up, up[i]))
                l_switch.append(True)
                l_switch_to.append('up')
                l_trend_down.append(dn[i])  # не смысла смотреть min, точно знаем что цена пробила
                l_trend_up.append(max(prev_trend_up, up[i]))
                trend_now = 'up'
        else:
            # значит был тренд вверх
            if prev_trend <= close:
                # пробития тренда не было, значит тренд остается
                l_trend.append(max(prev_trend, up[i]))
                l_switch_to.append('')
                l_switch.append(False)
                l_trend_up.append(max(prev_trend, up[i]))
                if prev_trend_down < close:
                    l_trend_down.append(dn[i])
                else:
                    l_trend_down.append(min(prev_trend_down, dn[i]))
            else:
                # тренд пробит, меняем линию
                l_trend.append(min(prev_trend_down, dn[i]))
                l_switch.append(True)
                l_switch_to.append('down')
                l_trend_up.append(up[i])  # не смысла смотреть max, точно знаем что цена пробила
                l_trend_down.append(min(prev_trend_down, dn[i]))
                trend_now = 'down'

    spread_df['trend'] = pd.DataFrame(l_trend)
    spread_df['trend_up'] = pd.DataFrame(l_trend_up)
    spread_df['trend_down'] = pd.DataFrame(l_trend_down)
    spread_df['switch'] = pd.DataFrame(l_switch)
    spread_df['switch_to'] = pd.DataFrame(l_switch_to)
    # start3 = datetime.datetime.now()
    # print(f'time1={start2-start1}, and time2={start3 - start2}')
    return spread_df


def zscore_calculating(df, lookback):

    if lookback > len(df):
        lookback = len(df)

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


def get_max_deviation_from_sma(df, lookback):
    # Принцип расчета:
    # 1. кол. правильных треугольников (изменение уровня дна < 30% от высоты) >30 на 2000
    # 1.1 неправильные считаем отдельно.
    # 2. кол отфильтрованных по времени (если > полураспада - не учитываем в расчетах) <=2
    # 3. кол отфильтрованных по маленькому проценту (< bb-1 (bb-1>0.6%) - не учитываем)
    # 4. средний и макс вынос на правильных треугольниках

    # norm_dev = list()
    # norm_dev = pd.DataFrame(columns=['close', 'sma', 'From', 'To', 'len', 'max_deviation'])
    full_dev = pd.DataFrame()
    # добавим данные для расчета
    # df["sma"] = df["close"].rolling(window=lookback, min_periods=1).mean()
    df['bb_up'], df['sma'], df['bb_down'] = talib.BBANDS(df.close, lookback, 1, 1, 0)
    df["signal"] = 0.0
    df["signal"] = np.where(df["close"] > df["sma"], 1.0, 0.0)
    df["position"] = df["signal"].diff()
    df = df.iloc[lookback:]

    # получим таблицу с временами пересечения SMA
    from_time = df[(df["position"] == 1) | (df["position"] == -1)]
    from_time = from_time.iloc[:-1]
    from_time = from_time.reset_index()
    from_time.drop(["open", "high", "low", "close", "sma", "signal", "position", "bb_up", "bb_down"], axis=1, inplace=True)
    from_time.rename(columns={"startTime": "From"}, inplace=True)
    from_time.rename(columns={"time": "FromTimestm"}, inplace=True)

    to_time = df[(df["position"] == 1) | (df["position"] == -1)]
    to_time = to_time.iloc[1:]
    to_time = to_time.reset_index()
    to_time.drop(["open", "high", "low", "close", "sma", "signal", "position", "bb_up", "bb_down"],
                 axis=1, inplace=True)
    to_time.rename(columns={"startTime": "To"}, inplace=True)
    to_time.rename(columns={"time": "ToTimestm"}, inplace=True)

    concated_from_to_time = pd.concat([from_time, to_time], axis=1)
    concated_from_to_time.drop(["index"], axis=1, inplace=True)

    # max_deviations_between_crossovers = pd.DataFrame()
    # abnormal_count = 0
    # low_count = 0
    # hl_count = 0
    halflive = lookback/3*2
    for _, row in concated_from_to_time.iterrows():
        # df_slice = df.loc[row["FromTimestm"]:row["ToTimestm"]]
        df_slice = df[(df.time >= row['FromTimestm']) & (df.time <= row['ToTimestm'])]

        first_cross = df_slice.iloc[0]['sma']
        bb_up = df_slice.iloc[0]['bb_up']
        bb_down = df_slice.iloc[0]['bb_down']
        # index_of_max_deviation = (df_slice['close'] - df_slice['sma']).abs().idxmax()  # откл от текущего знач sma
        index_of_max_deviation = (df_slice['close'] - first_cross).abs().idxmax()  # индекс строки с макс отклонением
        row_with_max_deviation = df_slice.loc[[index_of_max_deviation], ['close', 'sma']]  # строка с макс отклонением

        max_dev = row_with_max_deviation
        max_dev["From"] = row["From"]
        max_dev["To"] = row["To"]
        max_dev["len"] = len(df_slice)

        # %откл от текущего знач sma
        # max_dev["max_deviation"] = (max_dev["close"] - max_dev["sma"])/max_dev["sma"]*100

        # %откл от первого пересечения sma
        max_dev["max_deviation"] = abs((max_dev["close"] - first_cross) / first_cross * 100)
        max_deviation = max_dev.iloc[0]["max_deviation"]

        if bb_down < max_dev.iloc[0]["close"] < bb_up:
            max_dev["low_count"] = 1
            max_dev["norm_count"] = 0
            max_dev["abnormal_count"] = 0
            max_dev["hl_count"] = 0
        else:
            # рассчитаем изменение дна треугольника
            last_cross = df_slice.iloc[len(df_slice)-1]['sma']
            base_div = abs((last_cross - first_cross) / first_cross * 100)
            if base_div <= max_deviation/3:
                max_dev["low_count"] = 0
                max_dev["norm_count"] = 1
                max_dev["abnormal_count"] = 0
                max_dev["hl_count"] = 0
            else:
                max_dev["low_count"] = 0
                max_dev["norm_count"] = 1
                max_dev["abnormal_count"] = 1
                max_dev["hl_count"] = 0

        if len(df_slice) > halflive:
            max_dev["hl_count"] = 1

        full_dev = pd.concat([full_dev, max_dev], axis=0, ignore_index=True)

    return full_dev
