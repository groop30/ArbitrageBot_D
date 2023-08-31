# Основной уклон в данной версии - на ручную торговлю
# Стратегии задаются вручную, робот лишь следит что бы не пропустить вход/выход

import key
import datetime
import pandas as pd
import schedule
import time
import bin_utils as modul
from os import path
import ccxt
import talib

binance = ccxt.binanceusdm({
    'enableRateLimit': True,
    'apiKey': key.binanceAPI,
    'secret': key.binanceSecretAPI,
})

pd.options.mode.chained_assignment = None
tf_5m = 5 * 60
tf_1h = 60 * 60
tf_5m_str = '5m'
pos_size = 50.0
now_hour = datetime.datetime.now().hour
connection = modul.connect_to_sqlalchemy_binance()
use_sql_for_report = True
alerts = []


def open_position(action, pair, coin1, coin2, strategy, going_to, l_price, p_size, lookback,
                  stop, limit, up, down, l_time):
    if action == "signal":
        if pair not in alerts:
            modul.send_message_to_telegram(
                f'{pair}- сигнал на вход {going_to}, стратегия {strategy}, цена={l_price}')
            alerts.append(pair)
    elif action == "trade":
        print(f'{coin1}/{coin2}-открываем позицию {strategy},{going_to}, цена={l_price}, в {l_time}')
        modul.open_pair_position(connection, coin1, coin2, going_to, p_size, lookback, stop,
                                 limit, strategy, up, down)


def h1_permission(df_1h, no_stop=True):

    denied = 'trade'
    last_1h = df_1h.tail(2)
    # По тф 1Н смотрим только последнюю полностью сформированную свечу
    open_1h, high_1h, low_1h, close_1h, bb_up_1h, bb_down_1h = (
        last_1h[c].to_numpy()[0]
        for c in last_1h
        if c in ["open", "high", "low", "close", "bb_up", "bb_down"]
    )
    last_price = last_1h.iloc[1]['close']
    # смотрим, что бы последняя свеча была полностью за пределами канала, тогда блокируем действия
    if bb_down_1h > high_1h:
        denied = 'no_trade'
        if no_stop:
            return denied

        # если последняя свеча вне канала - смотрим, есть ли условие для стопа
        # условием будет, если цена зайдет НИЖЕ open/close полностью закрытой свечи
        if open_1h > close_1h:
            # свеча падающая, так как уже внизу, то опасность пойти еще ниже, лой не смотрим
            stop_level = close_1h
        else:
            # Свеча растущая, возможна ситуация когда начали отрастать.
            # Что бы не отстопило случайным движением, смотрим расстояние между open-close
            diff_body = close_1h - open_1h
            diff_body_per = diff_body / open_1h * 100
            if diff_body_per < 0.5:
                # если тело свечи меньше пол процента - то за стоп берем лой
                stop_level = low_1h
            else:
                stop_level = open_1h
        # если последняя цена опустилась ниже уровня стопа, то закрываем сделку.
        if last_price < stop_level:
            denied = 'stop'
    elif bb_up_1h < low_1h:
        denied = 'no_trade'
        if no_stop:
            return denied
        # если последняя свеча вне канала - смотрим, есть ли условие для стопа
        # условием будет, если цена зайдет ВЫШЕ open/close полностью закрытой свечи
        if open_1h < close_1h:
            # свеча растущая, так как цена уже вверху, то опасность пойти еще выше, хай не смотрим
            stop_level = close_1h
        else:
            # Свеча падающая, возможна ситуация когда начали снижаться.
            # Что бы не отстопило случайным движением, смотрим расстояние между open-close
            diff_body = open_1h - close_1h
            diff_body_per = diff_body / open_1h * 100
            if diff_body_per < 0.5:
                # если тело свечи меньше пол процента - то за стоп берем high
                stop_level = high_1h
            else:
                stop_level = open_1h
        # если последняя цена поднялась ВЫШЕ уровня стопа, то закрываем сделку.
        if last_price > stop_level:
            denied = 'stop'

    return denied


def update_pnl():
    close_df = pd.DataFrame()
    if use_sql_for_report:
        close_table = modul.create_close_table(connection)
        query = close_table.select()
        with connection.connect() as conn:
            close_df = pd.read_sql(sql=query, con=conn)
    else:
        filepath_close = r'.\reports\bin_to_close.csv'
        if path.exists(filepath_close):
            close_df = pd.read_csv(filepath_close, sep="\t")

    # анализируем открытые позиции, ищем ситуации для усреднения или для закрытия
    for index in range(len(close_df)):
        # получим данные строки
        coin1_id, coin2_id, pair, coin1, coin2, going_to, op_price, c1_op_price, c2_op_price = (
            close_df[c].to_numpy()[index]
            for c in close_df
            if c in ["coin1_id", "coin2_id", "pair", "coin1", "coin2", "going_to", "price", "c1_op_price", "c2_op_price"]
        )
        last_price1, last_price2, l_price = modul.get_last_spread_price(coin1, coin2, connection)

        if going_to == 'UP':
            if (c1_op_price is not None and c1_op_price > 0.0) and (c2_op_price is not None and c2_op_price > 0.0):
                coin1_res_perc = (last_price1 - c1_op_price) / c1_op_price * 100
                coin2_res_perc = (c2_op_price - last_price2) / c2_op_price * 100
                result_per = coin1_res_perc + coin2_res_perc
            else:
                result = l_price - op_price
                result_per = round(result / op_price * 100, 3)
        else:
            if (c1_op_price is not None and c1_op_price > 0.0) and (c2_op_price is not None and c2_op_price > 0.0):
                coin1_res_perc = (c1_op_price - last_price1) / c1_op_price * 100
                coin2_res_perc = (last_price2 - c2_op_price) / c2_op_price * 100
                result_per = coin1_res_perc + coin2_res_perc
            else:
                result = op_price - l_price
                result_per = round(result / op_price * 100, 3)

        modul.update_closedf(connection, coin1_id, 'pnl', str(result_per))


def cut_tails():

    tails1 = modul.find_lost_trades(connection, True, 'Binance')
    tails2 = modul.find_lost_trades(connection, True, 'Binance2')
    if len(tails1) > 0:
        tails1['exchange'] = 'Binance'
    if len(tails2) > 0:
        tails2['exchange'] = 'Binance2'
    tails = pd.concat([tails1, tails2], ignore_index=True)
    for index in range(len(tails)):
        coin = tails.iloc[index]['coin']
        size = tails.iloc[index]['diff']
        exchange = tails.iloc[index]['exchange']
        price_df = modul.get_last_price(coin)
        if size < 0:
            size = -size
            side = 'buy'
            l_price = price_df.iloc[0]['bid']
        else:
            side = 'sell'
            l_price = price_df.iloc[0]['ask']
        usd_size = l_price*size
        if usd_size < 10.0:
            # недостаточно для ордера, смотрим, последний или нет.
            plan_size = tails.iloc[index]['size']
            if plan_size == 0.0:
                # значит должно быть все закрыто, делаем ордер на увеличение суммы, что бы потом закрыть полностью
                # сразу не закрываем, даем полностью разместиться, закроем при следующем проходе.
                print(f'размещаем ордер на 11 долл, на увеличение суммы.')
                if side == 'buy':
                    modul.make_limit_order(coin=coin, amount=11.0, p_side='sell', size=0.0, exchange=exchange)
                else:
                    modul.make_limit_order(coin=coin, amount=11.0, p_side='buy', size=0.0, exchange=exchange)
        else:
            print(f'Закрываем хвост по {coin}, p_side={side}, size={size}')
            modul.make_limit_order(coin=coin, amount=0.0, p_side=side, size=size, exchange=exchange)


def check_for_open():

    # если баланс недостаточен - нет смысла смотреть дальше
    balance = modul.enough_balance()
    if balance < 10.0:
        print(f'недостаточно средств на балансе = {balance}')
        return False

    # получим таблицу уже открытых позиций
    if use_sql_for_report:
        close_table = modul.create_close_table(connection)
        check_table = modul.create_check_table(connection)
        query_close = close_table.select()
        query_check = check_table.select()
        with connection.connect() as conn:
            to_close_df = pd.read_sql(sql=query_close, con=conn)
            check_df = pd.read_sql(sql=query_check, con=conn)
            check_df = check_df[check_df['action'] == 'trade']
    else:
        check_df = pd.DataFrame(columns=["coin1", "coin2", "strategy", "down", "up", "action"])
        filepath_check = r'.\reports\bin_to_check.csv'
        if path.exists(filepath_check):
            if modul.read_file(filepath_check):
                check_df = pd.read_csv(filepath_check, sep="\t")

        filepath_close = r'.\reports\bin_to_close.csv'
        to_close_df = pd.read_csv(filepath_close, sep="\t")

    ########################################
    # проверяем каждую пару
    for index in range(len(check_df)):

        coin1, coin2, strategy, lookback, up, down, stop, action = (
            check_df[c].to_numpy()[index]
            for c in check_df
            if c in ["coin1", "coin2", "strategy", "lookback", "up", "down", "stop", "action"]
        )
        lookback = int(lookback)
        if lookback == 0:
            lookback = 50
        pair = f'{coin1}-{coin2}'
        # проверим отдельно по монетам, нет ли повторов
        opened_coin1 = to_close_df[(to_close_df['coin1'] == coin1)]
        opened_coin2 = to_close_df[(to_close_df['coin2'] == coin2)]
        # посмотрим, если ли уже есть открытые позиции
        opened_df = to_close_df[(to_close_df['coin1'] == coin1) & (to_close_df['coin2'] == coin2)]
        opened_positions = len(opened_df)
        if (len(opened_coin1) > 1 or len(opened_coin2) > 1) and opened_positions == 0:
            # значит есть другие пары с одной из монет, позиции не открываем.
            continue

        df = modul.calc_last_data(connection, coin1, coin2, lookback, tf_5m)

        # рассчитаем остальные данные
        l_time = datetime.datetime.now()
        last_row = df.tail(2)
        # получим данные последней строки
        l_price, l_zscore, l_bb_up, l_sma, l_bb_down = (
            last_row[c].to_numpy()[1]
            for c in last_row
            if c in ["close", "zscore", "bb_up", "sma", "bb_down"]
        )
        # получим данные предпоследней строки (последняя сформированная свеча)
        pre_high, pre_low, pre_price, pre_zscore, pre_bb_up, pre_sma, pre_bb_down = (
            last_row[c].to_numpy()[0]
            for c in last_row
            if c in ["high", "low", "close", "zscore", "bb_up", "sma", "bb_down"]
        )

        ########################################

        # подготовим данные для дальнейшего анализа
        l_diff = l_price - l_sma
        l_diff_per = l_diff / l_price * 100
        zsc_diff = l_zscore - pre_zscore

        if strategy == 'manual':
            if opened_positions > 0:
                continue
            if l_price < down:
                open_position(action, pair, coin1, coin2, strategy, 'UP', l_price, pos_size, lookback,
                              stop, True, up, down, l_time)
                # if action == "signal":
                #     if pair not in alerts:
                #         modul.send_message_to_telegram(f'{pair}- сигнал на вход UP, цена={l_price}')
                #         alerts.append(pair)
                # elif action == "trade":
                #     print(f'{coin1}/{coin2}-открываем позицию price_range,UP, цена={l_price}, в {l_time}')
                #     modul.open_pair_position(connection, coin1, coin2, "UP", pos_size, lookback, stop,
                #                              True, strategy, up, down)
            elif l_price > up:
                open_position(action, pair, coin1, coin2, strategy, 'DOWN', l_price, pos_size, lookback,
                              stop, True, up, down, l_time)
                # if action == "signal":
                #     if pair not in alerts:
                #         modul.send_message_to_telegram(f'{pair}- сигнал на вход DOWN, цена={l_price}')
                #         alerts.append(pair)
                # elif action == "trade":
                #     print(f'{coin1}/{coin2}-открываем позицию price_range,DOWN, цена={l_price}, в {l_time}')
                #     modul.open_pair_position(connection, coin1, coin2, "DOWN", pos_size, lookback, stop,
                #                              True, strategy, up, down)
            elif pair in alerts:
                if down < pre_price < up:
                    alerts.remove(pair)
        elif strategy == 'zscore':
            # блок условий для открытия позиций усреднения
            if opened_positions > 0:
                # усредняем позицию, пока что только если одна позиция.
                if opened_positions == 1:
                    # значит второй заказ еще не открыт, смотрим условия
                    op_price = opened_df.iloc[0]['price']
                    going_to = opened_df.iloc[0]['going_to']
                    stop = opened_df.iloc[0]['stop']
                    # для zscore 2 вход - при заходе внутрь уровня, если цена от пошла против нас > 3%
                    z_condition_1 = False
                    # пробую без второго условия
                    z_condition_2 = False
                    if going_to == 'UP':
                        price_diff = (op_price - l_price) / op_price * 100
                        if price_diff > 3:
                            z_condition_1 = True
                            # if (pre_zscore < go_up) & (l_zscore > go_up):
                            z_condition_2 = True
                        if z_condition_1 & z_condition_2:
                            # цена ушла ниже, и zsc пересек уровень снизу вверх, пора заходить
                            print(f'{coin1}/{coin2}-усредняем позицию zscore,UP, цена={l_price}, в {l_time}')
                            modul.open_pair_position(connection, coin1, coin2, "UP", pos_size, lookback, stop,
                                                     True, strategy, up, down)
                    elif going_to == 'DOWN':
                        price_diff = (l_price - op_price) / op_price * 100
                        if price_diff > 3:
                            z_condition_1 = True
                            # if (pre_zscore > go_down) & (l_zscore < go_down):
                            z_condition_2 = True
                        if z_condition_1 & z_condition_2:
                            # цена ушла выше, и zsc зашел обратно под уровень, пора заходить
                            print(f'{coin1}/{coin2}-усредняем позицию zscore,DOWN, цена={l_price}, в {l_time}')
                            modul.open_pair_position(connection, coin1, coin2, "DOWN", pos_size, lookback, stop,
                                                     True, strategy, up, down)

            else:
                # Если плановый профит меньше 0.6%, то не открываем сделку, т.к. все съест комиссия и проскальзывание
                if -0.6 < l_diff_per < 0.6:
                    continue
                # для zscore вход при начале уменьшения zsc, если он уже зашел за уровень
                if l_zscore < down:
                    if zsc_diff > 0.2:
                        # zsc начал увеличиваться, пора заходить
                        open_position(action, pair, coin1, coin2, strategy, 'UP', l_price, pos_size, lookback,
                                      stop, True, up, down, l_time)
                elif l_zscore > up:
                    if zsc_diff < -0.2:
                        # zsc начал уменьшаться, пора заходить
                        open_position(action, pair, coin1, coin2, strategy, 'DOWN', l_price, pos_size, lookback,
                                      stop, True, up, down, l_time)
        elif strategy == 'bb1':
            if opened_positions > 0:
                continue
            # Если плановый профит меньше 0.6%, то не открываем сделку, т.к. все съест комиссия и проскальзывание
            if -0.6 < l_diff_per < 0.6:
                continue
            # для ББ1 вход - на пересечении линии наружу,
            if l_price < l_bb_down:
                open_position(action, pair, coin1, coin2, strategy, 'UP', l_price, pos_size, lookback,
                              stop, True, up, down, l_time)
            elif l_price > l_bb_up:
                open_position(action, pair, coin1, coin2, strategy, 'DOWN', l_price, pos_size, lookback,
                              stop, True, up, down, l_time)
        elif strategy == 'bb2':
            if opened_positions > 0:
                continue
            # Для ББ2 вход - на первом заходе внутрь 3-го СКО
            # Если плановый профит меньше 0.6%, то не открываем сделку, т.к. все съест комиссия и проскальзывание
            df['bb_up_1'], df['sma_1'], df['bb_down_1'] = talib.BBANDS(df.close, lookback, 1, 1, 0)
            # df['bb_up_4'], df['sma_4'], df['bb_down_4'] = talib.BBANDS(df.close, lookback, 4.2, 4.2, 0)
            l_row = df.tail(1)
            # получим данные последней строки
            bb_up_1, bb_down_1 = (
                l_row[c].to_numpy()[0]
                for c in l_row
                if c in ["bb_up_1", "bb_down_1"]
            )

            if l_price > l_bb_down and pre_price < pre_bb_down:
                l_diff_down = bb_down_1 - l_price
                l_diff_per_down = l_diff_down / l_price * 100
                if l_diff_per_down < 0.6:
                    continue
                open_position(action, pair, coin1, coin2, strategy, 'UP', l_price, pos_size, lookback,
                              stop, True, up, down, l_time)

            elif l_price < l_bb_up and pre_price > pre_bb_up:
                l_diff_up = l_price - bb_up_1
                l_diff_per_up = l_diff_up / l_price * 100
                if l_diff_per_up < 0.6:
                    continue
                open_position(action, pair, coin1, coin2, strategy, 'DOWN', l_price, pos_size, lookback,
                              stop, True, up, down, l_time)

            elif pair in alerts:
                if l_bb_down < pre_price < l_bb_up:
                    alerts.remove(pair)
        elif strategy == 'grid_1':
            # Используется одна линия ББ(период 1000) с первым отклонением. Вход при пересечении ББ-1,
            # Следующие входы на расстоянии Х% от цены входа.
            # Тейк при пересечении SMA. Не входим при потенциале сделки менее 0.5%

            df['bb_up'], df['sma'], df['bb_down'] = talib.BBANDS(df.close, lookback, 1, 1, 0)
            _, _, time_to_opposite = modul.check_for_touch_bb(df, lookback, 1)
            if time_to_opposite > lookback / 2:
                continue

            # получим данные последней строки
            l_row = df.tail(1)
            bb_up_1 = l_row.iloc[0]["bb_up"]
            bb_down_1 = l_row.iloc[0]["bb_down"]
            stop = 0.0
            step = 0.02
            grid_rows = 5
            if opened_positions == 0:
                # Проверяем на условие первого входа
                if l_price > bb_up_1:
                    if l_diff_per > 0.5:
                        # открываем позицию в шорт
                        open_position(action, pair, coin1, coin2, strategy, 'DOWN', l_price, pos_size, lookback,
                                      stop, True, up, down, l_time)

                elif l_price < bb_down_1:
                    if l_diff_per < -0.5:
                        # открываем позицию в long
                        open_position(action, pair, coin1, coin2, strategy, 'UP', l_price, pos_size, lookback,
                                      stop, True, up, down, l_time)

            else:
                last_opened = opened_df.tail(1)
                l_op_price = last_opened.iloc[0]['price']
                going_to = last_opened.iloc[0]['going_to']
                # Если не все уровни открыты, смотрим не пора ли открыть новый
                if going_to == 'DOWN':
                    next_level = l_op_price + l_op_price * step
                    if l_price > next_level and opened_positions < grid_rows:
                        # открываем позицию в шорт
                        open_position(action, pair, coin1, coin2, strategy, going_to, l_price, pos_size, lookback,
                                      stop, True, up, down, l_time)

                else:
                    next_level = l_op_price - l_op_price * step
                    if l_price < next_level and opened_positions < grid_rows:
                        open_position(action, pair, coin1, coin2, strategy, going_to, l_price, pos_size, lookback,
                                      stop, True, up, down, l_time)
        elif strategy == 'bb3_atr':
            df['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=5)
            df['sma_atr'] = df['atr'].rolling(window=lookback, min_periods=1).mean()
            # Calculate ATR percentage
            df['atr_percentage'] = (df['atr'] / df['close']) * 100
            last_atr = df.tail(5)
            # получим данные последней строки
            atr_max = last_atr['atr'].max()
            atr_coeff = atr_max / last_atr.iloc[4]['sma_atr']
            if (pre_price < l_sma < l_price) | (pre_price > l_sma > l_price):
                # bars_before_open = 0 Начинаем отсчет свечей от момента пересечения SMA
                # Проверим, есть ли пересечение сма, если нам нужно его дождаться
                if stop > 0.0:
                    modul.update_check_df(connection, pair, "stop", 0.0)
            elif stop == 0.0:
                if opened_positions == 0:
                    if pre_price > l_bb_up > l_price and atr_coeff > 2.0:
                        if l_diff_per > 0.5:
                            # открываем позицию в шорт
                            open_position(action, pair, coin1, coin2, strategy, 'DOWN', l_price, pos_size, lookback,
                                          stop, True, up, down, l_time)
                    elif l_price > l_bb_down > pre_price and atr_coeff > 2.0:
                        if l_diff_per < -0.5:
                            # открываем позицию в лонг
                            open_position(action, pair, coin1, coin2, strategy, 'UP', l_price, pos_size, lookback,
                                          stop, True, up, down, l_time)


def open_one_time_positions():

    # если баланс недостаточен - нет смысла смотреть дальше
    balance = modul.enough_balance()
    if balance < 10.0:
        print(f'недостаточно средств на балансе ={balance}')
        return False

    filepath_open = r'.\reports\to_open.csv'

    if path.exists(filepath_open):
        open_df = pd.read_csv(filepath_open, sep="\t")
        new_df = open_df
        for index in range(len(open_df)):
            coin1, coin2, going_to, amount, strategy, lookback, op_price, take_price, stop = (
                open_df[c].to_numpy()[index]
                for c in open_df
                if c in ["coin1", "coin2", "going_to", "amount", "strategy", "lookback", "from", "to", "stop"]
            )

            _, _, l_price = modul.get_last_spread_price(coin1, coin2)
            if strategy == 'range':
                if (going_to == 'UP' and l_price < op_price) or (going_to == 'DOWN' and l_price > op_price):
                    modul.open_pair_position(connection, coin1, coin2, going_to, amount,
                                             lookback, stop, True, strategy, op_price, take_price)
                    print(f'{coin1}/{coin2}-открываем разовую позицию {strategy},{going_to}, цена={l_price}')
                    # убираем пару из таблицы
                    new_df = new_df[(new_df['coin1'] != coin1) & (new_df['coin2'] != coin2)]
            else:
                # пока для других стратегий условие не смотрим, сразу открываем
                modul.open_pair_position(connection, coin1, coin2, going_to, amount, lookback, stop,
                                         True, strategy, op_price, take_price)
                print(f'{coin1}/{coin2}-открываем разовую позицию {strategy},{going_to}, цена={l_price}')
                # убираем пару из таблицы
                new_df = new_df[(new_df['coin1'] != coin1) & (new_df['coin2'] != coin2)]

        new_df.to_csv(filepath_open, index=False, sep="\t")


# основная процедура
def check_for_close():
    close_df = pd.DataFrame()
    if use_sql_for_report:
        close_table = modul.create_close_table(connection)
        query = close_table.select()
        with connection.connect() as conn:
            close_df = pd.read_sql(sql=query, con=conn)
    else:
        filepath_close = r'.\reports\bin_to_close.csv'
        if path.exists(filepath_close):
            close_df = pd.read_csv(filepath_close, sep="\t")

    # анализируем открытые позиции, ищем ситуации для усреднения или для закрытия
    for index in range(len(close_df)):
        # получим данные строки
        coin1_id, coin2_id, pair, coin1, coin2, going_to, op_price, stop,  size1, size2, strategy, lookback, up, down, exchange = (
            close_df[c].to_numpy()[index]
            for c in close_df
            if c in ["coin1_id", "coin2_id", "pair", "coin1", "coin2", "going_to", "price", "stop",
                     "size1", "size2", "strategy", "lookback", "up", "down", "exchange"]
        )
        lookback = int(lookback)
        if lookback == 0:
            lookback = 50
        df = modul.calc_last_data(connection, coin1, coin2, lookback, tf_5m)
        _, _, l_price = modul.get_last_spread_price(coin1, coin2, connection)
        # подготовим остальные данные
        last_row = df.tail(1)
        l_sma = last_row.iloc[0]['sma']
        # создаем строку с данными
        new_row = pd.DataFrame({
            'coin1': [coin1],
            'coin2': [coin2],
            'going_to': [going_to],
            'cl_price': [round(l_price, 6)],
            'stop': [round(stop, 6)],
        },
            index=None)
        ########################################
        # блок условий для закрытия позиций
        if strategy == 'manual':
            if going_to == 'UP':
                if (up < l_price and up != 0.0) or l_price < stop:
                    modul.close_pair_position(connection, coin1_id, coin2_id, coin1, coin2,
                                              size1, size2, l_price, new_row)
            elif going_to == 'DOWN':
                if (down > l_price and down != 0.0) or (l_price > stop != 0.0):
                    modul.close_pair_position(connection, coin1_id, coin2_id, coin1, coin2,
                                              size1, size2, l_price, new_row)
        elif strategy == 'bb2':
            df['bb_up_1'], df['sma_1'], df['bb_down_1'] = talib.BBANDS(df.close, lookback, 1, 1, 0)
            # df['bb_up_4'], df['sma_4'], df['bb_down_4'] = talib.BBANDS(df.close, lookback, 4.2, 4.2, 0)
            l_row = df.tail(1)
            # получим данные последней строки
            bb_up_1, bb_down_1 = (
                l_row[c].to_numpy()[0]
                for c in l_row
                if c in ["bb_up_1", "bb_down_1"]
            )
            if going_to == 'UP':
                if (l_price > bb_down_1) | (stop != 0.0 and l_price < stop):
                    modul.close_pair_position(connection, coin1_id, coin2_id, coin1, coin2,
                                              size1, size2, l_price, new_row)
            elif going_to == 'DOWN':
                if (l_price < bb_up_1) | (stop != 0.0 and l_price > stop):
                    modul.close_pair_position(connection, coin1_id, coin2_id, coin1, coin2,
                                              size1, size2, l_price, new_row)
        elif strategy == 'grid_1':
            df['bb_up_1'], df['sma_1'], df['bb_down_1'] = talib.BBANDS(df.close, lookback, 1, 1, 0)
            # получим данные последней строки
            l_row = df.tail(1)
            bb_up_1 = l_row.iloc[0]["bb_up_1"]
            bb_down_1 = l_row.iloc[0]["bb_down_1"]
            stop_per = 15.0
            opened_df = close_df[(close_df['pair'] == pair)]
            opened_positions = len(opened_df)
            # Если не все уровни открыты, смотрим не пора ли открыть новый
            if going_to == 'DOWN':
                if l_price < l_sma and opened_positions <= 2:
                    modul.close_pair_position(connection, coin1_id, coin2_id, coin1, coin2, size1, size2, l_price,
                                              new_row, exchange)
                elif l_price < bb_down_1 and opened_positions > 2:
                    modul.close_pair_position(connection, coin1_id, coin2_id, coin1, coin2, size1, size2, l_price,
                                              new_row, exchange)
                elif l_price > stop != 0.0:
                    modul.close_pair_position(connection, coin1_id, coin2_id, coin1, coin2, size1, size2, l_price,
                                              new_row, exchange)
                    modul.update_check_df(connection, pair, 'action', 'waiting')  # Отключаем пару от торгов
                # elif (l_price - op_price)/op_price*100 > stop_per:
                elif (l_price - l_sma) / l_sma * 100 > stop_per:
                    new_row['stop'] = stop_per
                    modul.close_pair_position(connection, coin1_id, coin2_id, coin1, coin2, size1, size2, l_price,
                                              new_row, exchange)
                    modul.update_check_df(connection, pair, 'action', 'waiting')  # Отключаем пару от торгов
            else:
                if l_price > l_sma and opened_positions <= 2:
                    modul.close_pair_position(connection, coin1_id, coin2_id, coin1, coin2, size1, size2, l_price,
                                              new_row, exchange)
                elif l_price > bb_up_1 and opened_positions > 2:
                    modul.close_pair_position(connection, coin1_id, coin2_id, coin1, coin2, size1, size2, l_price,
                                              new_row, exchange)
                elif l_price < stop != 0.0:
                    modul.close_pair_position(connection, coin1_id, coin2_id, coin1, coin2, size1, size2, l_price,
                                              new_row, exchange)
                    modul.update_check_df(connection, pair, 'action', 'waiting')  # Отключаем пару от торгов
                # elif (op_price-l_price)/op_price*100 > stop_per:
                elif (l_sma - l_price) / l_sma * 100 > stop_per:
                    new_row['stop'] = stop_per
                    modul.close_pair_position(connection, coin1_id, coin2_id, coin1, coin2, size1, size2, l_price,
                                              new_row, exchange)
                    modul.update_check_df(connection, pair, 'action', 'waiting')  # Отключаем пару от торгов
        elif strategy == 'bb3_atr':
            df['bb_up_4'], _, df['bb_down_4'] = talib.BBANDS(df.close, lookback, 4.2, 4.2, 0)
            l_row = df.tail(1)
            bb_up_4 = l_row.iloc[0]["bb_up_4"]
            bb_down_4 = l_row.iloc[0]["bb_down_4"]
            if going_to == 'UP':
                if (l_price > l_sma) | (stop != 0.0 and l_price < stop):
                    modul.close_pair_position(connection, coin1_id, coin2_id, coin1, coin2,
                                              size1, size2, l_price, new_row, True, exchange)
                elif l_price < bb_down_4:
                    new_row['stop'] = bb_down_4
                    modul.close_pair_position(connection, coin1_id, coin2_id, coin1, coin2,
                                              size1, size2, l_price, new_row, True, exchange)
            elif going_to == 'DOWN':
                if (l_price < l_sma) | (stop != 0.0 and l_price > stop):
                    modul.close_pair_position(connection, coin1_id, coin2_id, coin1, coin2,
                                              size1, size2, l_price, new_row, True, exchange)
                elif l_price > bb_up_4:
                    new_row['stop'] = bb_down_4
                    modul.close_pair_position(connection, coin1_id, coin2_id, coin1, coin2,
                                              size1, size2, l_price, new_row, True, exchange)
        elif strategy == 'st_dev':
            df = modul.rolling_st_dev_channels(df, lookback, 1.5)
            # получим данные последней строки
            l_row = df.tail(1)
            l_center = l_row.iloc[0]["line_center"]
            if going_to == 'DOWN':
                if l_price < l_center:
                    modul.close_pair_position(connection, coin1_id, coin2_id, coin1, coin2, size1, size2, l_price,
                                              new_row, exchange)
                elif l_price > stop != 0.0:
                    modul.close_pair_position(connection, coin1_id, coin2_id, coin1, coin2, size1, size2, l_price,
                                              new_row, exchange)
                    modul.update_check_df(connection, pair, 'action', 'waiting')  # Отключаем пару от торгов
            else:
                if l_price > l_center:
                    modul.close_pair_position(connection, coin1_id, coin2_id, coin1, coin2, size1, size2, l_price,
                                              new_row, exchange)
                elif l_price < stop != 0.0:
                    modul.close_pair_position(connection, coin1_id, coin2_id, coin1, coin2, size1, size2, l_price,
                                              new_row, exchange)
                    modul.update_check_df(connection, pair, 'action', 'waiting')  # Отключаем пару от торгов
        else:
            # Общее правило закрытия, используется, если для стратегии не прописано отдельных условий выше
            if going_to == 'UP':
                if (l_price > l_sma) | (stop != 0.0 and l_price < stop):
                    modul.close_pair_position(connection, coin1_id, coin2_id, coin1, coin2,
                                              size1, size2, l_price, new_row, True, exchange)
            elif going_to == 'DOWN':
                if (l_price < l_sma) | (stop != 0.0 and l_price > stop):
                    modul.close_pair_position(connection, coin1_id, coin2_id, coin1, coin2,
                                              size1, size2, l_price, new_row, True, exchange)


# schedule.every(15).seconds.do(check_for_close)
schedule.every(10).minutes.do(cut_tails)
schedule.every(5).minutes.do(update_pnl)
# # # schedule.every(1).minutes.do(open_one_time_positions)
while True:
    schedule.run_pending()
    check_for_open()
    check_for_close()
    time.sleep(15)
# update_pnl()
