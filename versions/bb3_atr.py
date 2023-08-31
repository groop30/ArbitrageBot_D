import datetime
import pandas as pd
import time
import bin_utils as modul
from os import path
import talib

pd.options.mode.chained_assignment = None
tf_5m = 5 * 60
tf_5m_str = '5m'
pos_size = 100.0
connection = modul.connect_to_sqlalchemy_binance()
alerts = []


def open_position(action, pair, coin1, coin2, strategy, going_to, l_price, lookback,
                  stop, limit, up, down, this_time):
    # if action == "signal":
    if pair not in alerts:
        modul.send_message_to_telegram(
            f'BINANCE:{coin1}.P/BINANCE:{coin2}.P- сигнал на вход {going_to}, цена={l_price}, '
            f'https://www.tradingview.com/chart/?symbol=BINANCE:{coin1}.P/BINANCE:{coin2}.P')
        alerts.append(pair)
    # else:
    print(f'{coin1}/{coin2}-открываем позицию {strategy},{going_to}, цена={l_price}, в {this_time}')
    modul.open_pair_position(connection, coin1, coin2, going_to, pos_size, lookback, stop,
                             limit, strategy, up, down, True, 'Binance2')


def check_for_open():
    # если баланс недостаточен - нет смысла смотреть дальше
    balance = modul.enough_balance("Binance2")
    if balance < 10.0:
        print(f'недостаточно средств на балансе = {balance}')
        return False

    check_df = pd.DataFrame(columns=["coin1", "coin2", "strategy", "down", "up", "action"])
    filepath_check = r'.\screening\3_bb3_atr.csv'
    if path.exists(filepath_check):
        if modul.read_file(filepath_check):
            check_df = pd.read_csv(filepath_check, sep="\t")

        close_table = modul.create_close_table(connection)
        query_close = close_table.select()
        with connection.connect() as conn:
            to_close_df = pd.read_sql(sql=query_close, con=conn)
            to_close_df = to_close_df[to_close_df['strategy'] == 'bb3_atr']

        ########################################
        # проверяем каждую пару
        for index in range(len(check_df)):
            pair = check_df.iloc[index]['pair']
            coin1, coin2 = modul.pair_to_coins(pair)

            # проверим отдельно по монетам, нет ли повторов
            opened_coin1 = to_close_df[(to_close_df['coin1'] == coin1)]
            opened_coin2 = to_close_df[(to_close_df['coin2'] == coin2)]
            # посмотрим, если ли уже есть открытые позиции
            opened_df = to_close_df[(to_close_df['coin1'] == coin1) & (to_close_df['coin2'] == coin2)]
            opened_positions = len(opened_df)
            if (len(opened_coin1) > 1 or len(opened_coin2) > 1) and opened_positions == 0:
                # значит есть другие пары с одной из монет, позиции не открываем.
                continue

            lookback = 240
            action = "signal"
            strategy = "bb3_atr"

            df = modul.calc_last_data(connection, coin1, coin2, lookback, tf_5m)

            # рассчитаем остальные данные
            l_time = datetime.datetime.now()
            last_row = df.tail(2)
            # получим данные последней строки
            l_price, l_bb_up, l_sma, l_bb_down = (
                last_row[c].to_numpy()[1]
                for c in last_row
                if c in ["close", "bb_up", "sma", "bb_down"]
            )
            # получим данные предпоследней строки (последняя сформированная свеча)
            pre_high, pre_low, pre_price, ppre_bb_up, pre_sma, pre_bb_down = (
                last_row[c].to_numpy()[0]
                for c in last_row
                if c in ["high", "low", "close", "bb_up", "sma", "bb_down"]
            )

            ########################################

            # подготовим данные для дальнейшего анализа
            l_diff = l_price - l_sma
            l_diff_per = l_diff / l_price * 100

            df['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=5)
            df['sma_atr'] = df['atr'].rolling(window=lookback, min_periods=1).mean()
            # Calculate ATR percentage
            df['atr_percentage'] = (df['atr'] / df['close']) * 100
            last_atr = df.tail(5)

            # получим данные последней строки
            atr_max = last_atr['atr'].max()
            atr_coeff = atr_max / last_atr.iloc[4]['sma_atr']

            if (pre_price < l_sma < l_price) | (pre_price > l_sma > l_price):

                # Проверим, есть ли пересечение сма, если нам нужно его дождаться
                if pair in alerts:
                    alerts.remove(pair)
                    # modul.update_check_df(connection, pair, "stop", 0.0)
            elif pair not in alerts:
                if opened_positions == 0:
                    if pre_price > l_bb_up > l_price and atr_coeff > 2.0:
                        if l_diff_per > 0.5:
                            # открываем позицию в шорт
                            open_position(action, pair, coin1, coin2, strategy, 'DOWN', l_price, lookback,
                                          0.0, True, 0.0, 0.0, l_time)
                    elif l_price > l_bb_down > pre_price and atr_coeff > 2.0:
                        if l_diff_per < -0.5:
                            # открываем позицию в лонг
                            open_position(action, pair, coin1, coin2, strategy, 'UP', l_price, lookback,
                                          0.0, True, 0.0, 0.0, l_time)


while True:
    check_for_open()
    time.sleep(15)
