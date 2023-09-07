import pandas as pd
import numpy as np
import talib


def williams_fractals(df, n):
    high_prices = df['high']
    low_prices = df['low']
    ddf_high = pd.DataFrame()
    ddf_high['high'] = high_prices
    ddf_low = pd.DataFrame()
    ddf_low['low'] = low_prices
    mask_list = []
    for num_col in range(1,n+1):
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
    # ddf_high['max_value'] = ddf_high.max(axis=1)
    # ddf_high['pivot_high_test'] = (ddf_high['max_value'] == ddf_high['high'])


    # ddf_high['one_left'] = high_prices.shift(1)
    # ddf_high['two_left'] = high_prices.shift(2)
    # ddf_high['one_right'] = high_prices.shift(-1)
    # ddf_high['two_right'] = high_prices.shift(-2)
    # ddf_high['pivot_high'] = (ddf_high['high'] >= ddf_high['one_left']) & (ddf_high['high'] >= ddf_high['two_left']) \
    #                          & (ddf_high['high'] > ddf_high['one_right']) & (ddf_high['high'] > ddf_high['two_right'])

    # ddf_low['one_left'] = low_prices.shift(1)
    # ddf_low['two_left'] = low_prices.shift(2)
    # ddf_low['one_right'] = low_prices.shift(-1)
    # ddf_low['two_right'] = low_prices.shift(-2)
    # ddf_low['pivot_low'] = (ddf_low['low'] <= ddf_low['one_left']) & (ddf_low['low'] <= ddf_low['two_left']) \
    #                          & (ddf_low['low'] < ddf_low['one_right']) & (ddf_low['low'] < ddf_low['two_right'])

    # return up_fractals, down_fractals
    return ddf_high['pivot_high'], ddf_low['pivot_low']


def pivot_point_supertrend(spread_df, pp_prd, atr_factor, atr_prd):

    # Get high and low prices
    high = spread_df['high']
    low = spread_df['low']
    close = spread_df['close']

    # Calculate pivots
    pivot_high_series, pivot_low_series = williams_fractals(spread_df, pp_prd)
    pivot_high_values = np.where(pivot_high_series, spread_df['high'], False)
    pivot_low_values = np.where(pivot_low_series, spread_df['low'], False)
    all_pivots = np.where(pivot_high_values, pivot_high_values, pivot_low_values)
    pd.Series(all_pivots).replace(0, np.nan, inplace=True)

    # filled_center = pd.Series(center).fillna(method='ffill')
    # spread_df['center'] = pd.Series(filled_center).rolling(2).apply(lambda x: (x[-1] * 2 + x[0]) / 3, raw=True)
    # center_df = pd.DataFrame({'center': np.nan}, index=range(len(all_pivots)))
    # pivot_series = pd.Series(all_pivots)
    # pivot_mask = ~pivot_series.isna()
    # center_df.loc[pp_prd+1:, 'center'] = (center_df['center'].shift(1) * 2 + pivot_series[pivot_mask].shift(pp_prd)) / 3
    #
    # data = {'A': [1, 2, 3, 4, 5]}  # Replace this with your actual data
    # # Create a DataFrame
    # df = pd.DataFrame(data)
    # # Calculate column B
    # df['B'] = 0
    # df['B'] = (df['B'].shift(1).fillna(0) * 2 + df['A']) / 3

    spread_df['pivots'] = all_pivots
    spread_df['center'] = np.nan
    for i in range((pp_prd+1), len(all_pivots)):
        pivot_now = all_pivots[i - pp_prd]
        if not pd.isna(pivot_now):
            if not pd.isna(spread_df.iloc[i-1]['center']):
                spread_df.at[i, 'center'] = (spread_df.iloc[i - 1]['center'] * 2 + pivot_now) / 3
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

    # # Get the trend - подумать, как сделать без цикла...
    # trend = talib.MAX(close, timeperiod=prd)
    # trend = np.where(close < talib.MAX(close.shift(), timeperiod=prd), -1, trend)
    # spread_df['trend_down'] = np.where(close > dn, -1, dn)
    # spread_df['trend_down'] = np.where((spread_df['trend_down'].shift() < dn)&(spread_df['trend_down']!=-1), spread_df['trend_down'], dn)
    # trend_up = pd.Series(np.where(close < up , -1, up))
    # spread_df['trend_up'] = pd.Series(np.where(trend_up.shift() > up, trend_up, up))
    spread_df['trend'] = np.nan
    spread_df['trend_up'] = np.nan
    spread_df['trend_down'] = np.nan
    spread_df['switch'] = False
    spread_df['switch_to'] = ''
    now_trend = ''
    for i in range(atr_prd, len(spread_df)):
        prev_trend = spread_df.iloc[i - 1]['trend']
        prev_trend_down = spread_df.iloc[i - 1]['trend_down']
        prev_trend_up = spread_df.iloc[i-1]['trend_up']
        close = spread_df.iloc[i]['close']
        if pd.isna(prev_trend):
            if close < up[i]:
                spread_df.at[i, 'trend'] = dn[i]
                spread_df.at[i, 'switch_to'] = 'down'
                # ####################
                spread_df.at[i, 'trend_down'] = dn[i]
                spread_df.at[i, 'trend_up'] = up[i]
                now_trend = 'down'
            else:
                spread_df.at[i, 'trend'] = up[i]
                spread_df.at[i, 'switch_to'] = 'up'
                # ####################
                spread_df.at[i, 'trend_down'] = dn[i]
                spread_df.at[i, 'trend_up'] = up[i]
                now_trend = 'up'
            spread_df.at[i, 'switch'] = True

        else:
            # смотрим предыдущие значения, что бы понять, какой был тренд
            if now_trend == 'down':
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

            else:
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

    return spread_df