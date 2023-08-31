import key
import ccxt
import pandas as pd
import datetime
from pathlib import Path
from os import path
from typing import Optional
from binance.client import Client
from binance.enums import HistoricalKlinesType
import talib
import statsmodels.api as stat
import statsmodels.tsa.stattools as ts
import numpy as np
from statsmodels.tsa.vector_ar.vecm import coint_johansen
import plotly.graph_objects as go

binance = ccxt.binanceusdm({
    'enableRateLimit': True,
    'apiKey': key.binanceAPI,
    'secret': key.binanceSecretAPI,
})
binance_api = Client(
    api_key=key.binanceAPI,
    api_secret=key.binanceSecretAPI
)

pd.options.mode.chained_assignment = None

tf_5m = 5 * 60
tf_1h = 60 * 60
tf_5m_str = '5m'


def get_last_price(coin):
    res = binance.fetch_bids_asks([coin])
    df = pd.DataFrame(res)
    df = df.T
    return df


def get_position(order_id, asset):
    res = binance.fetch_order(order_id, symbol=asset)
    df = pd.DataFrame([res])
    return df


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


def get_delta_adf(data1, data2):
    # Perform ADF test on the closing prices of fetched data
    # на тестах большой разницы между тестом АДФ на обычную цену НЕ показал
    result = stat.OLS(data1, data2).fit()
    ct = ts.adfuller(result.resid, maxlag=1)
    return ct[1]


# рассчитать стационарности ряда
def stationarity(a):
    a = np.ravel(a)
    stat_res = ts.adfuller(a, maxlag=1)
    stat_coeff = stat_res[1]
    return stat_coeff


# рассчитать коэф коинтеграции
def cointegration(a, b):
    coint_coeff = ts.coint(a, b, maxlag=1)
    p_value = coint_coeff[1]
    # t_stat = coint_coeff[0]
    # per_10 = coint_coeff[2][2]
    # # Checking co-integration
    # if t_stat <= per_10 and p_value <= 0.05:
    #     return p_value
    # else:
    return p_value


def cointegration_johansen(a, b):
    # doing an inner join to make sure dates coincide and there are no NaNs
    # inner join requires distinct column names
    main_df = pd.merge(a, b, how='inner', on='startTime')
    # get rid of extra columns but keep the date index
    main_df.drop(labels=['open_x', 'open_y', 'high_x', 'high_y', 'low_x',
                         'low_y', 'time_y'],
                 axis=1,
                 inplace=True)
    main_df.rename(columns={'close_y': 'y', 'close_x': 'x'}, inplace=True)

    # The second and third parameters indicate constant term, with a lag of 1.
    result = coint_johansen(main_df, 0, 1)

    return result


# use Engle-Granger two-step method to test cointegration
# the underlying method is straight forward and easy to implement
# a more important thing is the method is invented by the mentor of my mentor!!!
# the latest statsmodels package should ve included johansen test which is more common
# check sm.tsa.var.vecm.coint_johansen
# the malaise of two-step is the order of the cointegration
# unlike johansen test, two-step method can only detect the first order
# check the following material for further details
# https://warwick.ac.uk/fac/soc/economics/staff/gboero/personal/hand2_cointeg.pdf
def eg_method(x, y, show_summary=False):
    # step 1
    # estimate long run equilibrium
    model1 = stat.OLS(y, stat.add_constant(x)).fit()
    epsilon = model1.resid

    if show_summary:
        print('\nStep 1\n')
        print(model1.summary())

    # check p value of augmented dickey fuller test
    # if p value is no larger than 5%, stationary test is passed
    if stat.tsa.stattools.adfuller(epsilon)[1] > 0.05:
        return False, model1

    # take first order difference of X and Y plus the lagged residual from step 1
    x_dif = stat.add_constant(pd.concat([x.diff(), epsilon.shift(1)], axis=1).dropna())
    y_dif = y.diff().dropna()

    # step 2
    # estimate error correction model
    model2 = stat.OLS(y_dif, x_dif).fit()

    if show_summary:
        print('\nStep 2\n')
        print(model2.summary())

    # adjustment coefficient must be negative
    if list(model2.params)[-1] > 0:
        return False, model1
    else:
        return True, model1


def request_history(asset, tf_str, s_start, lookforward):
    """
    :param asset:
    :param tf_str: таймфрейм текстом!!!!
    :param s_start: момент старта в формате timestamp
    :param lookforward: количество свечей в указанном таймфрейме (не в секундах!)
    :return:
    """
    # res = binance.fetch_ohlcv(asset, tf_str, s_start, limit=lookforward)
    end_time = s_start + lookforward*tf_5m
    s_start = int(s_start*1000)
    end_time = int(end_time*1000)
    # TODO - проверить запрос!
    # res = binance_api.futures_historical_klines
    res = binance_api.get_historical_klines(
        symbol=asset,
        interval=tf_str,
        start_str=s_start,
        end_str=end_time,
        limit=lookforward,
        klines_type=HistoricalKlinesType.FUTURES
    )
    df = pd.DataFrame(res)
    if len(df) > 0:
        df.drop(df.columns[[5, 6, 7, 8, 9, 10, 11]], axis=1, inplace=True)
        df.columns = ['time', 'open', 'high', 'low', 'close']
        df['startTime'] = df['time'].map(lambda x: datetime.datetime.fromtimestamp(x / 1000))
        df['startTime'] = df['startTime'].astype(str)
        df['open'] = df['open'].astype(float)
        df['high'] = df['high'].astype(float)
        df['low'] = df['low'].astype(float)
        df['close'] = df['close'].astype(float)
        df = df[['startTime', 'time', 'open', 'high', 'low', 'close']]
        df = prepare_dataframe(df=df, timestamp_field="startTime", asc=False)

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


def make_spread_df(df_coin1, df_coin2, last_to_end=False, tf=tf_5m):
    """
    :param df_coin1: датафреймы, полученные из get_history_price
    :param df_coin2: датафреймы, полученные из get_history_price
    :param last_to_end: порядок сортировки. при True - последние данные будут в конце, и наоборот
    :return: датафрейм отношения df_coin1/df_coin2
    """
    # теперь сделать итоговую таблицу
    main_df = pd.merge(df_coin1, df_coin2, how='inner', on='startTime')
    main_df['open'] = main_df['open_x']/main_df['open_y']
    main_df['high'] = main_df['high_x'] / main_df['low_y']
    main_df['low'] = main_df['low_x'] / main_df['high_y']
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

    if 'time_x' in main_df.columns:
        main_df = main_df.rename(columns={"time_x": "time"})

    # main_df.index = pd.DatetimeIndex(main_df['startTime'])
    main_df.sort_values(
        by='startTime',
        ascending=last_to_end,
        inplace=True,
        ignore_index=True,
    )

    if tf != tf_5m:
        main_df = convert_to_tf(main_df, tf)

    return main_df


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
    df.reset_index(inplace=True)
    return df


def get_history_price(asset, start, end, tf=tf_5m):
    """
    :param asset:
    :param start:  момент с которого нужны данные в формате timestamp
    :param end:  момент окончания в формате timestamp
    :param tf: количество секунд в нужном таймфрейме
    :return:
    """
    # на данный момент возвращать только 5м таймфрейм
    if tf != tf_5m:
        # history_df = convert_to_tf(history_df, tf)
        tf = tf_5m

    tf_str = tf_5m_str
    s_start = start
    filename = f"{asset}.csv"
    filepath = Path("files", filename)
    history_df = pd.DataFrame()
    concat_needed = True
    if path.exists(filepath):
        try:
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
    if concat_needed and tf == tf_5m and lookforward != 0:
        # запрашиваем из базы только, если объединение нужно
        try:
            df = request_history(asset, tf_str, s_start, lookforward)
            df = pd.concat([history_df, df], ignore_index=True)

            if isinstance(df, pd.DataFrame) and len(df) > 0:
                df = prepare_dataframe(df=df, timestamp_field="startTime", asc=False)
                df.to_csv(filepath, columns=["startTime", "time", "open", "high", "low", "close"], index=False,
                          sep="\t")
        except Exception as ex:
            print(asset, tf, s_start, end, ex)

    df = df[(df['time'] >= start * 1000) & (df['time'] <= end * 1000)]
    return df


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


def get_coin_min_size(coin):

    min_size = 0.01
    try:
        # get_symbol_info - работает только со спотом. пришлось переписать
        # info = binance_api.get_symbol_info(coin)
        # filters = info["filters"]
        info = binance_api.futures_exchange_info()
        futures_list = info["symbols"]
        f_df = pd.DataFrame(futures_list)
        coin_row = f_df[f_df['pair'] == coin]
        filters = coin_row.iloc[0]['filters']
        for i in filters:
            if i["filterType"] == "LOT_SIZE":
                min_size = i['minQty']
                break
    except Exception as error:
        min_size = 0.01
        print(f'{coin} выдает ошибку в информации по символу - {error}')

    return float(min_size)


def find_stop_loss(df, direction, strategy='zscore'):

    stop = 0.0
    # подготовим остальные данные
    last_row = df.tail(1)
    l_sma = last_row.iloc[0]['sma']
    l_price = last_row.iloc[0]['close']

    if direction == 'UP':
        # отберем те свечи, которые за диапазоном
        # df['cross'] = np.where((df['zscore'] > 0) & (df['zsc_shift'] <= 0)),
        #                                'zero',
        #                                df['zsc_shift'])
        to_sma = l_sma - l_price
        # находим минимальную цену, с момента выхода из диапазона
        stop_price = df['low'].min()
    else:
        # находим максимальную цену, с момента выхода из диапазона
        stop_price = df['high'].max()

    return stop


def calculate_historical_profit(hist_df, pair, strategy, sma, up, down):

    hist_df = zscore_calculating(hist_df, sma)
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

    filepath_total = r'.\screening\4_optimize_result.csv'
    if path.exists(filepath_total):
        # добавим к имеющимся парам для отслеживания новые
        total_df = pd.read_csv(filepath_total, sep="\t")
        total_df = pd.concat([total_df, total_row], ignore_index=True)
        total_df.to_csv(filepath_total, index=False, sep="\t")
    else:
        total_row.to_csv(filepath_total, index=False, sep="\t")


def make_graph(df):
    fig = go.Figure(data=[go.Candlestick(x=df['startTime'],
                                         open=df['open'],
                                         high=df['high'],
                                         low=df['low'],
                                         close=df['close'])])
    # fig.add_trace(go.Indicator(df["sma"]))
    fig.show()
