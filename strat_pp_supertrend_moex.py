import datetime
import pandas as pd
import time
import bin_utils as modul
import indicators as ind
import alor as alor_modul

pd.options.mode.chained_assignment = None
tf_5m = 5 * 60
tf_5m_str = '5m'
pos_size = 20.0
alerts = []
connection = modul.connect_to_sqlalchemy_binance()
headers = alor_modul.autorization()


def send_alert(coin, going_to, l_price):

    if coin not in alerts:
        modul.send_message_to_telegram(
            f'{coin}- сигнал на вход {going_to}, цена={l_price}, '
            f'https://www.tradingview.com/chart/?symbol=MOEX:{coin}', 3)
        alerts.append(coin)


def check_for_open():

    # all_futures = modul.get_all_futures()
    all_futures =['CRZ3', 'SIZ3', 'EuZ3', 'NGV3', 'BRV3', 'EDZ3', 'GDZ3', 'RIZ3', 'MMZ3', 'SVZ3', 'GAZP', 'SBER', 'LKOH',
                  'TRNFP', 'VTBR', 'ROLO', 'MGNT', 'MOEX', 'GMKN', 'YNDX', 'ROSN', 'CHMF', 'TATN', 'ALRS', 'SNGSP', 'NVTK',
                  'QIWI', 'MTLR', 'NLMK', 'PIKK', 'MAGN', 'PHOR', 'NMTP', 'IRAO', 'MTSS', 'SBERP', 'HYDR', 'RUAL', 'AGRO', 'SNGS', 'AFKS']

    for i in range(len(all_futures)):
        future = all_futures[i]
        end_date = datetime.datetime.now().timestamp()
        # end_date = datetime.datetime(2023, 9, 15, 10, 30, 0).timestamp()
        start_date = end_date - 1800*tf_5m
        history_df = alor_modul.get_sql_history_price(future, connection, start_date, end_date, headers)
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
                        if (l_price - trend_start) / trend_start > 0.004:
                            send_alert(future, 'UP', l_price)

            else:
                # переключились на растущий тренд, смотрим два предыдущих
                check_df = check_df[check_df['switch_to'] == 'down']
                trend_start = check_df.iloc[-1]['trend']
                if len(check_df) > 1:
                    if trend_start < check_df.iloc[-2]['trend']:
                        # if (trend_start - history_df.iloc[-2]['trend']) / history_df.iloc[-2]['trend'] > 0.005:
                        #     if l_price < trend_start:
                        if (trend_start - l_price) / l_price > 0.004:
                            send_alert(future, 'DOWN', l_price)

        ########################################
        # Проверим, удалять ли из списка?
        if (future in alerts) and (not l_switch):
            alerts.remove(future)


while True:
    check_for_open()
    time.sleep(240)
