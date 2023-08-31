from pathlib import Path
import pandas as pd
from os import path
import bin_utils as modul
import time
import datetime
import key
from binance.client import Client
from binance.enums import HistoricalKlinesType

tf_5m = 5 * 60
tf_1m = 60
tf_5m_str = '5m'
tf_1m_str = '1m'
binance_api = Client(
    api_key=key.binanceAPI,
    api_secret=key.binanceSecretAPI
)

#'LINKUSDT', 'AAVEUSDT', 'BNBUSDT', 'ATOMUSDT', 'ETHUSDT',
            # 'ADAUSDT', 'XRPUSDT', 'SOLUSDT', 'AVAXUSDT', 'TRXUSDT'
#             '1000SHIBUSDT', 'ALGOUSDT', '1000XECUSDT', 'MANAUSDT', 'REEFUSDT', 'RENUSDT', 'SPELLUSDT',
#               'AXSUSDT', 'PEOPLEUSDT', 'LINAUSDT', 'WAVESUSDT', 'HNTUSDT', 'ICPUSDT', 'NKNUSDT', 'APEUSDT', 'SANDUSDT',
main_list = ['TRXUSDT', 'ADAUSDT', 'ATOMUSDT', 'SOLUSDT', 'AAVEUSDT', 'AVAXUSDT']


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


def request_history(asset, tf_str, s_start, lookforward, sql=False, tf=tf_5m):
    """
    :param asset:
    :param tf_str: таймфрейм текстом!!!!
    :param s_start: момент старта в формате timestamp
    :param lookforward: количество свечей в указанном таймфрейме (не в секундах!)
    :return:
    """
    # res = binance.fetch_ohlcv(asset, tf_str, s_start, limit=lookforward)
    rest_candels = lookforward
    df = pd.DataFrame()
    while rest_candels > 0:
        if rest_candels > 1000:
            end_time = s_start + 1000*tf
            s_start = int(s_start*1000)
            end_time = int(end_time*1000)
            look = 1000
        else:
            end_time = s_start + rest_candels*tf
            s_start = int(s_start*1000)
            end_time = int(end_time*1000)
            look = rest_candels
        # TODO - проверить запрос!
        # res = binance_api.futures_historical_klines
        res = binance_api.get_historical_klines(
            symbol=asset,
            interval=tf_str,
            start_str=s_start,
            end_str=end_time,
            limit=look,
            klines_type=HistoricalKlinesType.FUTURES
        )
        s_start = end_time/1000
        rest_candels = rest_candels - 1000
        part_df = pd.DataFrame(res)
        df = pd.concat([df, part_df], ignore_index=True)

    if len(df) > 0:
        df.drop(df.columns[[6, 7, 8, 9, 10, 11]], axis=1, inplace=True)
        df.columns = ['time', 'open', 'high', 'low', 'close', 'volume']
        df['startTime'] = df['time'].map(lambda x: datetime.datetime.fromtimestamp(x / 1000))
        if not sql:
            df['startTime'] = df['startTime'].astype(str)
        df['open'] = df['open'].astype(float)
        df['high'] = df['high'].astype(float)
        df['low'] = df['low'].astype(float)
        df['close'] = df['close'].astype(float)
        df = df[['startTime', 'time', 'open', 'high', 'low', 'close', 'volume']]
        if sql:
            df = prepare_dataframe(df=df, timestamp_field="startTime", asc=True)
        else:
            df = prepare_dataframe(df=df, timestamp_field="startTime", asc=False)
    return df


def get_history_price(asset, start, end, tf=tf_5m):
    """
    :param asset:
    :param start:  момент с которого нужны данные в формате timestamp
    :param end:  момент окончания в формате timestamp
    :param tf: количество секунд в нужном таймфрейме
    :return:
    """
    # на данный момент возвращать только 5-м таймфрейм
    if tf == tf_1m:
        tf_str = tf_1m_str
    else:
        tf_str = tf_5m_str


    s_start = start
    filename = f"{asset}.csv"
    filepath = Path("files", filename)
    history_df = pd.DataFrame(columns=['startTime', 'time', 'open', 'high', 'low', 'close'])
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
    if concat_needed and lookforward != 0:
        # запрашиваем из базы только, если объединение нужно
        try:
            df = request_history(asset, tf_str, s_start, lookforward, False, tf)
            df = pd.concat([history_df, df], ignore_index=True)

            if isinstance(df, pd.DataFrame) and len(df) > 0:
                df = prepare_dataframe(df=df, timestamp_field="startTime", asc=False)
                df.to_csv(filepath, columns=["startTime", "time", "open", "high", "low", "close", 'volume'], index=False,
                          sep="\t")
        except Exception as ex:
            print(asset, tf, s_start, end, ex)

    df = df[(df['time'] >= start * 1000) & (df['time'] <= end * 1000)]
    return df


def fill_deep_history_from(start_time):
    """

    :param start_time:
    :return:
    """

    # all_coins = modul.get_all_futures()
    # all_futures = all_coins.id.tolist()
    # all_futures = ['DUSKUSDT', 'LINAUSDT']
    max_time = float(1000*tf_1m)
    final_time = datetime.datetime.now().timestamp()
    for future in main_list:
        s_time = start_time
        print(f'Заполняем данные по {future}')
        filename = f"{future}.csv"
        filepath = Path("files", filename)
        coin_df = pd.DataFrame()
        if path.exists(filepath):
            coin_df = pd.read_csv(filepath, sep="\t")
            if isinstance(coin_df, pd.DataFrame) and len(coin_df) > 0:
                coin_df = prepare_dataframe(df=coin_df, timestamp_field="startTime", asc=False)
                while final_time > s_time:
                    time_gap = final_time - s_time
                    if time_gap > max_time:
                        time_gap = max_time
                    end_time = s_time + time_gap
                    get_history_price(future, s_time, end_time, tf_1m)
                    s_time = s_time + time_gap
                    time.sleep(1)
        else:
            end_time = s_time + max_time
            get_history_price(future, s_time, end_time, tf_1m)
            time.sleep(1)


def reformating_klines_df():
    for future in main_list:
        print(f'Обрабатываем файл {future}')
        filename = f"{future}.csv"
        filepath = Path("files", filename)
        if path.exists(filepath):
            coin_df = pd.read_csv(filepath, sep="\t", parse_dates=['startTime'])
            # coin_df.drop(labels=['time'], axis=1, inplace=True)
            # coin_df['pd_date'] = pd.to_datetime(coin_df['startTime'])
            coin_df['date'] = coin_df['startTime'].dt.strftime("%Y%m%d")
            coin_df['time'] = coin_df['startTime'].dt.time
            coin_df['ticker'] = future
            coin_df['per'] = 1
            # coin_df['volume'] = 1
            coin_df = modul.prepare_dataframe(df=coin_df, timestamp_field="startTime", asc=True)
            # coin_df.drop(labels=['startTime'], axis=1, inplace=True)
            filename = f"{future}_1.csv"
            filepath = Path("files", filename)
            coin_df.to_csv(filepath, columns=["ticker", "per", "date", "time", "open", "high", "low", "close", "volume"], index=False,
                          sep=",")


# блок первичного заполнения исторических данных
start_t = datetime.datetime(2021, 1, 1, 0, 0).timestamp()
# fill_deep_history_from(start_t)
reformating_klines_df()