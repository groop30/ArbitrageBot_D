import datetime
import pandas as pd
import time
import bin_utils as modul
import indicators as ind
# import numpy as np

pd.options.mode.chained_assignment = None
tf_5m = 5 * 60
tf_5m_str = '5m'
pos_size = 20.0
connection = modul.connect_to_sqlalchemy_binance()
alerts = []


def open_position(action, coin, strategy, going_to, l_price, lookback, stop, limit, this_time):
    if "signal" in action:
        if coin not in alerts:
            modul.send_message_to_telegram(
                f'BINANCE:{coin}.P- сигнал на вход {going_to}, цена={l_price}, '
                f'https://www.tradingview.com/chart/?symbol=BINANCE:{coin}.P', 2)
            alerts.append(coin)
    if "trade" in action:
        print(f'{coin}-открываем позицию {strategy},{going_to}, цена={l_price}, в {this_time}')
        modul.open_single_position(connection, coin, going_to, pos_size, lookback, stop, limit, strategy, True, 'Binance2')


def check_for_open():
    # если баланс недостаточен - нет смысла смотреть дальше
    balance = modul.enough_balance("Binance2")
    action = "signal, trade"
    if balance < 3.0:
        # print(f'недостаточно средств на балансе = {balance}')
        # return False
        action = "signal"

    # all_futures = modul.get_all_futures()
    all_futures = ['1000LUNCUSDT', '1000PEPEUSDT', '1000SHIBUSDT', '1INCHUSDT', 'AAVEUSDT',
     'ADAUSDT', 'AGIXUSDT', 'ALGOUSDT', 'AMBUSDT', 'ANTUSDT', 'APEUSDT',
     'API3USDT', 'APTUSDT', 'ARBUSDT', 'ARPAUSDT', 'ARUSDT', 'ASTRUSDT', 'ATOMUSDT',
     'AVAXUSDT', 'AXSUSDT', 'BAKEUSDT', 'BANDUSDT', 'BCHUSDT',
     'BLZUSDT', 'BNBUSDT', 'BTCUSDT', 'C98USDT', 'CELOUSDT', 'CFXUSDT', 'CHZUSDT',
     'COMPUSDT', 'CRVUSDT', 'CYBERUSDT', 'DASHUSDT', 'DODOXUSDT', 'DOGEUSDT', 'DOTUSDT', 'DYDXUSDT',
     'EOSUSDT', 'ETCUSDT', 'FETUSDT', 'FILUSDT', 'FTMUSDT',
     'GALAUSDT', 'GALUSDT', 'GMTUSDT', 'GRTUSDT', 'GTCUSDT', 'HBARUSDT', 'ICPUSDT', 'IMXUSDT', 'INJUSDT',
     'KAVAUSDT', 'KNCUSDT', 'LDOUSDT', 'LEVERUSDT', 'LINAUSDT', 'LINKUSDT',
     'LPTUSDT', 'LTCUSDT', 'LUNA2USDT', 'MAGICUSDT', 'MANAUSDT', 'MASKUSDT', 'MATICUSDT',
     'MKRUSDT', 'MTLUSDT', 'NEARUSDT', 'OCEANUSDT',
     'OPUSDT', 'PENDLEUSDT', 'PEOPLEUSDT', 'RDNTUSDT', 'REEFUSDT', 'RNDRUSDT', 'RUNEUSDT',
     'SANDUSDT', 'SEIUSDT', 'SFPUSDT', 'SNXUSDT', 'SOLUSDT', 'STMXUSDT',
     'STXUSDT', 'SUIUSDT', 'SXPUSDT', 'THETAUSDT', 'TOMOUSDT', 'TRUUSDT', 'TRXUSDT',
     'UNFIUSDT', 'UNIUSDT', 'WAVESUSDT', 'WLDUSDT', 'WOOUSDT', 'XMRUSDT', 'XRPUSDT', 'XVGUSDT', 'YGGUSDT']


    close_table = modul.create_close_table(connection)
    query_close = close_table.select()

    strategy = 'pp_supertrend'
    for i in range(len(all_futures)):
        # future = all_futures.iloc[i]["id"]
        future = all_futures[i]
        end_date = datetime.datetime.now().timestamp()
        # end_date = datetime.datetime(2023, 8, 28, 0, 30, 0).timestamp()
        start_date = end_date - 1800*tf_5m
        with connection.connect() as conn:
            to_close_df = pd.read_sql(sql=query_close, con=conn)
            to_close_df = to_close_df[to_close_df['strategy'] == strategy]

        # проверим отдельно по монетам, нет ли повторов
        opened_coin = to_close_df[(to_close_df['coin1'] == future)]
        if len(opened_coin) > 0:
            # значит есть другие позиции, не открываем.
            continue

        history_df = modul.get_sql_history_price(future, connection, start_date, end_date)
        history_df = modul.convert_to_tf(history_df, 900)  # 15 min timeframe

        history_df.sort_values(by='time', ascending=True, inplace=True, ignore_index=True)
        history_df = ind.pivot_point_supertrend(history_df, 2, 3, 10)

        last_row = history_df.tail(1)
        # получим данные последней строки
        l_time, l_price, l_trend, l_switch, l_switch_to = (
            last_row[c].to_numpy()[0]
            for c in last_row
            if c in ["startTime", "close", "trend", "switch", "switch_to"]
        )

        check_df = history_df.copy()

        # Версия 2.
        if l_switch:
            # Проверяем на условие первого входа
            if l_switch_to == 'down':
                # переключились на падающий тренд, смотрим два предыдущих
                check_df = check_df[check_df['switch_to'] == 'up']
                trend_start = check_df.iloc[-1]['trend']
                if len(check_df) > 1:
                    if trend_start > check_df.iloc[-2]['trend']:
                        # if (history_df.iloc[-2]['trend'] - trend_start) / trend_start > 0.005:
                        #     if l_price > trend_start:  # что бы после открытия сразу не отстопило
                        if (l_price - trend_start) / trend_start > 0.005:
                            open_position(action, future, strategy, 'UP', l_price, 0, round(trend_start, 7), True, l_time)

            else:
                # переключились на растущий тренд, смотрим два предыдущих
                check_df = check_df[check_df['switch_to'] == 'down']
                trend_start = check_df.iloc[-1]['trend']
                if len(check_df) > 1:
                    if trend_start < check_df.iloc[-2]['trend']:
                        # if (trend_start - history_df.iloc[-2]['trend']) / history_df.iloc[-2]['trend'] > 0.005:
                        #     if l_price < trend_start:
                        if (trend_start - l_price) / l_price > 0.005:
                            open_position(action, future, strategy, 'DOWN', l_price, 0, round(trend_start, 7), True, l_time)

        ########################################
        # Проверим, удалять ли из списка?
        if (future in alerts) and (not l_switch):
            alerts.remove(future)


while True:
    check_for_open()
    time.sleep(240)
