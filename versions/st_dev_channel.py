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
    # if pair not in alerts:
    #     modul.send_message_to_telegram(
    #         f'BINANCE:{coin1}.P/BINANCE:{coin2}.P- сигнал на вход {going_to}, цена={l_price}, '
    #         f'https://www.tradingview.com/chart/?symbol=BINANCE:{coin1}.P/BINANCE:{coin2}.P')
    #     alerts.append(pair)
    # else:
    print(f'{coin1}/{coin2}-открываем позицию {strategy},{going_to}, цена={l_price}, в {this_time}')
    modul.open_pair_position(connection, coin1, coin2, going_to, pos_size, lookback, stop,
                             limit, strategy, up, down, True, 'Binance')


def check_for_open():
    # если баланс недостаточен - нет смысла смотреть дальше
    balance = modul.enough_balance()
    if balance < 10.0:
        print(f'недостаточно средств на балансе = {balance}')
        return False

    check_df = pd.DataFrame(columns=["coin1", "coin2", "strategy", "down", "up", "action"])
    filepath_check = r'.\screening\bb_touch_result.csv'
    if path.exists(filepath_check):
        if modul.read_file(filepath_check):
            check_df = pd.read_csv(filepath_check, sep="\t")

        close_table = modul.create_close_table(connection)
        query_close = close_table.select()
        with connection.connect() as conn:
            to_close_df = pd.read_sql(sql=query_close, con=conn)
            to_close_df = to_close_df[to_close_df['strategy'] == 'st_dev']

        ########################################

        lookback = 1000
        std_dev = 1.5
        stop_per = 0.1
        action = "signal"
        strategy = "st_dev"

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

            l_time = datetime.datetime.now()
            end_time = l_time.timestamp()
            start_time = end_time - (lookback + 1) * tf_5m
            df_coin1 = modul.get_sql_history_price(coin1, connection, start_time, end_time)
            df_coin2 = modul.get_sql_history_price(coin2, connection, start_time, end_time)
            df = modul.make_spread_df(df_coin1, df_coin2, last_to_end=True, tf=tf_5m)
            if len(df) < lookback:
                # исторических данных не достаточно, пропускаем
                continue
            elif len(df) > lookback:
                df = df[-lookback:]

            df = modul.standard_deviation_channel(df, lookback, std_dev)
            # рассчитаем остальные данные

            last_row = df.tail(2)
            # получим данные последней строки
            l_price, l_up, l_center, l_down = (
                last_row[c].to_numpy()[1]
                for c in last_row
                if c in ["close", "line_up", "line_center", "line_down"]
            )
            # получим данные предпоследней строки (последняя сформированная свеча)
            pre_price, pre_up, pre_center, pre_down = (
                last_row[c].to_numpy()[0]
                for c in last_row
                if c in ["close", "line_up", "line_center", "line_down"]
            )

            ########################################

            # подготовим данные для дальнейшего анализа
            l_diff = l_price - l_center
            l_diff_per = l_diff / l_price * 100

            if pair in alerts:
                if (pre_price < l_center < l_price) | (pre_price > l_center > l_price):
                    alerts.remove(pair)
            else:
                # сначала смотрим условия для открытия позиции
                if opened_positions == 0:
                    if pre_price > l_up > l_price:
                        # if close > line_up:
                        if l_diff_per > 0.5:
                            # открываем позицию в шорт
                            stop = l_price + l_price*stop_per
                            open_position(action, pair, coin1, coin2, strategy, 'DOWN', l_price, lookback,
                                          stop, True, 0.0, 0.0, l_time)
                    elif l_price > l_down > pre_price:
                        # elif close < line_down:
                        if l_diff_per > 0.5:
                            # открываем позицию в лонг
                            stop = l_price - l_price * stop_per
                            open_position(action, pair, coin1, coin2, strategy, 'UP', l_price, lookback,
                                          stop, True, 0.0, 0.0, l_time)


while True:
    check_for_open()
    time.sleep(15)