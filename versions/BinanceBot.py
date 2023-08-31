import key
import datetime
import pandas as pd
import talib
from decimal import Decimal
import schedule
import time
import bin_utils as modul
from os import path
import ccxt

binance = ccxt.binanceusdm({
    'enableRateLimit': True,
    'apiKey': key.binanceAPI,
    'secret': key.binanceSecretAPI,
})

pd.options.mode.chained_assignment = None
tf_5m = 5 * 60
tf_1h = 60 * 60
tf_5m_str = '5m'
pos_size = 100.0
now_hour = datetime.datetime.now().hour
connection = modul.connect_to_sqlalchemy()
use_sql_for_report = True


def read_file(filename) -> bool:
    try:
        with open(file=filename, mode='r') as fh:
            fh.read()
        return True
    except Exception as error:
        print(f'ошибка прочтения файла {filename}, пробуем снова - {error}')
        time.sleep(3)
        read_file(filename=filename)


def cut_tails():

    tails = modul.find_lost_trades(connection, True)
    for index in range(len(tails)):
        coin = tails.iloc[index]['coin']
        size = tails.iloc[index]['diff']
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
                    make_limit_order(coin=coin, amount=11.0, p_side='sell', size=0.0)
                else:
                    make_limit_order(coin=coin, amount=11.0, p_side='buy', size=0.0)
        else:
            print(f'Закрываем хвост по {coin}, p_side={side}, size={size}')
            make_limit_order(coin=coin, amount=0.0, p_side=side, size=size)


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
            # свеча растущая. возможна ситуация когда начали отрастать.
            # что бы не отстопило случайным движением, смотрим расстояние между open-close
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
            # свеча падающая. возможна ситуация когда начали снижаться.
            # что бы не отстопило случайным движением, смотрим расстояние между open-close
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


def place_market_order(coin, p_size, p_side="buy"):
    """

    :param coin: название монеты
    :param p_size: размер позиции в единицах монеты!!! Не Доллары!
    :param p_side: направление ордера текстом
    :return:
    """

    try:
        o_result = binance.create_order(
            symbol=f"{coin}",
            side=p_side,
            price=0.0,
            amount=p_size,
            type='market'
        )
        return o_result
    except Exception as e:
        print(f'Error making order request: {e}')
        return None


# ##############################################
# блок управления лимитными ордерами
def make_limit_order(coin, amount, p_side, size=0.0):
    """

    :param coin: название монеты
    :param amount: размер позиции в долларах
    :param size: размер позиции в единицах монеты (ТОЛЬКО ДЛЯ ЗАКРЫТИЯ ПОЗИЦИИ)
    :param p_side: направление позиции
    :return:
    """

    # Запросить последнюю цену по бид/аск
    coin_df = modul.get_last_price(coin)
    if p_side == 'buy':
        l_price = coin_df.iloc[0]['bid']
    else:
        l_price = coin_df.iloc[0]['ask']

    if size == 0.0:
        # рассчитать размер позиции
        min_size1 = modul.get_coin_min_size(coin)
        if min_size1 >= 1.0:
            dec = 0
        else:
            dec = Decimal(str(min_size1)).as_tuple().exponent * (-1)
        p_size = round(amount / l_price, dec)
    else:
        # если закрываем позицию, то размер уже известен
        p_size = size

    # разместить ордер через place_limit_order
    res = place_limit_order(coin, l_price, p_size, p_side)
    if res is not None:
        res_df = pd.DataFrame([res])
        order_id = res_df.iloc[0]['id']
        res_df = manage_limit_order(order_id, coin, p_size, p_side, 1)
    else:
        res_df = pd.DataFrame()

    res_df['p_size'] = p_size

    return res_df


# открытие нового лимитного ордера
def place_limit_order(coin, p_price, p_size, p_side="buy"):

    # post_only = {'timeInForce': 'PostOnly'}
    try:
        o_result = binance.create_limit_order(
            symbol=f"{coin}",
            side=p_side,
            amount=p_size,
            price=p_price,
            # params=post_only
        )
        return o_result
    except Exception as e:
        print(f'Ошибка при размещении лимитного ордера по {coin} в {p_side}: {e}')
        return None


# отмена открытого ордера
def cancel_limit_order(order_id, coin):
    # Place an order
    try:
        o_result = binance.cancel_order(id=order_id, symbol=coin)
        return o_result
    except Exception as e:
        print(f'Ошибка при отмене лимитного ордера по {coin}: {e}')
        return None


# управление ордером до момента полного открытия позиции
def manage_limit_order(order_id, coin, p_size, p_side, count=1):

    time.sleep(1)
    # проверим, исполнен ли ордер
    res = binance.fetch_order(id=order_id, symbol=coin)

    result_df = pd.DataFrame([res])
    order_df = pd.DataFrame([res['info']])
    status = order_df.iloc[0]['status']
    full_qty = float(order_df.iloc[0]['origQty'])
    fill_qty = float(order_df.iloc[0]['executedQty'])
    od_price = float(order_df.iloc[0]['price'])
    # проверим исполнение ордера
    if status != 'FILLED':
        # если 'FILLED' - значит ордер исполнен, дальше не смотрим
        # если нет, смотрим нужно ли переместить или изменить размер
        new_size = full_qty - fill_qty
        was_changed = False
        coin_df = modul.get_last_price(coin)
        if p_side == 'buy':
            l_price = coin_df.iloc[0]['bid']
            if l_price > od_price:
                # цена ушла выше, нужно переставить ордер
                try:
                    res_modify = binance.edit_limit_order(
                        id=order_id,
                        symbol=coin,
                        side=p_side,
                        amount=new_size,
                        price=l_price
                    )
                    if res_modify is not None:
                        was_changed = True
                except Exception as error:
                    # скорее всего за время расчетов ордер успел исполниться, ничего не меняем.
                    print(f'лимитный ордер изменить не получилось - {error}')
        else:
            l_price = coin_df.iloc[0]['ask']
            if l_price < od_price:
                # цена ушла ниже, нужно переставить ордер
                try:
                    res_modify = binance.edit_limit_order(
                        id=order_id,
                        symbol=coin,
                        side=p_side,
                        amount=new_size,
                        price=l_price
                    )
                    if res_modify is not None:
                        was_changed = True
                except Exception as error:
                    # скорее всего за время расчетов ордер успел исполниться, ничего не меняем.
                    print(f'лимитный ордер изменить не получилось - {error}')
        if was_changed:
            res_df = pd.DataFrame([res_modify])
            order_id = res_df.iloc[0]['id']
            p_size = new_size

        time.sleep(1)
        if count < 4:
            count = count + 1
            res_df = manage_limit_order(order_id, coin, p_size, p_side, count)
        else:
            cancel_res = cancel_limit_order(order_id, coin)
            # если отменить не получилось, значит успел исполниться
            if cancel_res is not None:
                res_market = place_market_order(coin, p_size, p_side)
                res_df = pd.DataFrame([res_market])
            else:
                res_df = result_df
            # res_df = res_df.rename(columns={"orderId": "id"})
        return res_df
    else:
        return result_df


def save_close_changes(coin1_id):

    if use_sql_for_report:
        close_table = modul.create_close_table(connection)
        query = close_table.delete().where(close_table.columns.coin1_id == int(coin1_id))
        with connection.connect() as conn:
            conn.execute(query)
    else:
        filepath = r'.\reports\bin_to_close.csv'
        if path.exists(filepath):
            close_df = pd.read_csv(filepath, sep="\t")
            # убираем пару из таблицы
            close_df = close_df[close_df['coin1_id'] != coin1_id]
            close_df.to_csv(filepath, index=False, sep="\t")


def save_to_log(idd, row, new):
    filepath = r'.\reports\bin_to_log.csv'
    log_df = pd.DataFrame()
    if path.exists(filepath):
        log_df = pd.read_csv(filepath, sep="\t")

    if new:
        row = row.rename(columns={'price': 'op_price'})
        row['op_time'] = datetime.datetime.now()
        log_df = pd.concat([log_df, row], ignore_index=True)
    else:
        deal_df = log_df.loc[log_df['coin1_id'] == idd]
        if len(deal_df) > 0:
            ind_row = deal_df.index[0]
            cl_price = row.iloc[0]['cl_price']
            stop = row.iloc[0]['stop']
            log_df.at[ind_row, 'cl_price'] = cl_price
            log_df.at[ind_row, 'cl_time'] = datetime.datetime.now()
            log_df.at[ind_row, 'stop'] = stop
            op_price = log_df.iloc[ind_row]['op_price']
            going_to = log_df.iloc[ind_row]['going_to']
            if going_to == 'UP':
                result = cl_price - op_price
            else:
                result = op_price - cl_price
            result_per = result/op_price*100
            log_df.at[ind_row, 'result'] = round(result, 6)
            log_df.at[ind_row, 'result_perc'] = round(result_per, 3)
            per_no_commis = result_per - 0.16
            log_df.at[ind_row, 'per_no_commis'] = round(per_no_commis, 3)

    log_df.to_csv(filepath, index=False, sep="\t")


def close_position(order_id, coin, size=0.0, limit=False):
    pos_df = modul.get_position(order_id, coin)
    if len(pos_df) > 0:
        l_side = pos_df.iloc[0]['side']
        if size == 0.0:
            l_size = pos_df.iloc[0]['amount']
        else:
            l_size = size

        if l_side == 'buy':
            new_side = 'sell'
        else:
            new_side = 'buy'

        # теперь делаем обратный ордер
        if limit:
            result = make_limit_order(coin, 0.0, new_side, l_size)
        else:
            result = place_market_order(coin, l_size, new_side)
        return result
    else:
        print(f'не закрыта позиция по {coin}! Срочно закрыть вручную!!!')


def get_last_spread_price(coin1, coin2):
    # получим последнюю актуальную цену
    coin1_df = modul.get_last_price(coin1)
    coin2_df = modul.get_last_price(coin2)
    last_price1 = coin1_df.iloc[0]['bid']
    last_price2 = coin2_df.iloc[0]['bid']
    l_price = last_price1 / last_price2
    return l_price


def calc_last_data(coin1, coin2, lookback, tf):

    end_time = datetime.datetime.now().timestamp()
    start_time = datetime.datetime.now().timestamp() - lookback*tf - tf*50
    # df_coin1 = modul.get_history_price(coin1, start_time, end_time, tf)
    # df_coin2 = modul.get_history_price(coin2, start_time, end_time, tf)
    df_coin1 = modul.get_sql_history_price(coin1, connection, start_time, end_time)
    df_coin2 = modul.get_sql_history_price(coin2, connection, start_time, end_time)
    df = modul.make_spread_df(df_coin1, df_coin2, True, tf)
    if len(df_coin1) > lookback and len(df_coin2) > lookback:
        df = modul.zscore_calculating(df, lookback)
    else:
        print(f'недостаточно исторических данных по монете {coin1} или {coin2}')
        df['zscore'] = 0.0

    df['bb_up'], df['sma'], df['bb_down'] = talib.BBANDS(df.close, lookback, 4, 4, 0)

    return df


def check_for_open():

    # если баланс недостаточен - нет смысла смотреть дальше
    if not modul.enough_balance():
        print('недостаточно средств на балансе')
        return False

    check_df = pd.DataFrame(columns=["coin1", "coin2", "strategy", "go_up", "go_down"])
    filepath_check = r'.\reports\bin_to_check.csv'
    if path.exists(filepath_check):
        if read_file(filepath_check):
            check_df = pd.read_csv(filepath_check, sep="\t")

    # получим таблицу уже открытых позиций
    if use_sql_for_report:
        close_table = modul.create_close_table(connection)
        query = close_table.select()
        with connection.connect() as conn:
            to_close_df = pd.read_sql(sql=query, con=conn)
    else:
        filepath_close = r'.\reports\bin_to_close.csv'
        to_close_df = pd.read_csv(filepath_close, sep="\t")

    ########################################
    # проверяем каждую пару
    for index in range(len(check_df)):

        coin1, coin2, strategy, lookback, go_up, go_down = (
            check_df[c].to_numpy()[index]
            for c in check_df
            if c in ["coin1", "coin2", "strategy", "lookback", "go_up", "go_down"]
        )
        lookback = int(lookback)

        df = calc_last_data(coin1, coin2, lookback, tf_5m)
        # если разрешение на уровне Н1 не получено - сделок не открываем
        df_1h = calc_last_data(coin1, coin2, lookback, tf_1h)
        h1_denied = h1_permission(df_1h)
        if h1_denied != 'trade':
            continue

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

        # посмотрим, если ли уже есть открытые позиции
        opened_df = to_close_df[(to_close_df['coin1'] == coin1) & (to_close_df['coin2'] == coin2)]
        ########################################
        # блок условий для открытия позиций усреднения
        if len(opened_df) > 0:
            # усредняем позицию, пока что только если одна позиция.
            if len(opened_df) == 1:
                # значит второй заказ еще не открыт, смотрим условия
                op_price = opened_df.iloc[0]['price']
                going_to = opened_df.iloc[0]['going_to']
                stop = opened_df.iloc[0]['stop']

                if strategy == 'zscore':
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
                            open_pair_position(coin1, coin2, "UP", pos_size, lookback, stop,
                                               True, strategy, go_down, go_up)
                    elif going_to == 'DOWN':
                        price_diff = (l_price - op_price) / op_price * 100
                        if price_diff > 3:
                            z_condition_1 = True
                        # if (pre_zscore > go_down) & (l_zscore < go_down):
                            z_condition_2 = True
                        if z_condition_1 & z_condition_2:
                            # цена ушла выше, и zsc зашел обратно под уровень, пора заходить
                            print(f'{coin1}/{coin2}-усредняем позицию zscore,DOWN, цена={l_price}, в {l_time}')
                            open_pair_position(coin1, coin2, "DOWN", pos_size, lookback, stop,
                                               True, strategy, go_down, go_up)
                    continue
                elif strategy == 'bb1' or strategy == 'bb2':
                    # на обратном пересечении (внутрь), если от первого входа цена ушла > 3%
                    if going_to == 'UP':
                        price_diff = (op_price - l_price) / op_price * 100
                        if l_price > l_bb_down and pre_high < pre_bb_down and price_diff > 3:
                            print(f'{coin1}/{coin2}-открываем позицию bb2,UP, цена={l_price}, в {l_time}')
                            open_pair_position(coin1, coin2, "UP", pos_size, lookback, stop,
                                               True, strategy, go_down, go_up)
                    elif going_to == 'DOWN':
                        price_diff = (l_price - op_price) / op_price * 100
                        if l_price < l_bb_up and pre_low > pre_bb_up and price_diff > 3:
                            print(f'{coin1}/{coin2}-открываем позицию bb2,DOWN, цена={l_price}, в {l_time}')
                            open_pair_position(coin1, coin2, "DOWN", pos_size, lookback, stop,
                                               True, strategy, go_down, go_up)
                    continue
                else:
                    continue
            else:
                continue

        # подготовим данные для дальнейшего анализа
        l_diff = l_price - l_sma
        l_diff_per = l_diff / l_price * 100
        zsc_diff = l_zscore - pre_zscore

        # Если плановый профит меньше 0.6%, то не открываем сделку, т.к. все съест комиссия и проскальзывание
        if -0.6 < l_diff_per < 0.6:
            continue
        stop = 0.0
        if strategy == 'zscore':
            # для zscore вход при начале уменьшения zsc, если он уже зашел за уровень
            if l_zscore < go_up:
                if zsc_diff > 0.2:
                    # zsc начал увеличиваться, пора заходить
                    print(f'{coin1}/{coin2}-открываем позицию zscore,UP, цена={l_price}, в {l_time}')
                    # stop = modul.find_stop_loss(df, 'UP')
                    open_pair_position(coin1, coin2, "UP", pos_size, lookback, stop,
                                       True, strategy, go_down, go_up)
            elif l_zscore > go_down:
                if zsc_diff < -0.2:
                    # zsc начал уменьшаться, пора заходить
                    print(f'{coin1}/{coin2}-открываем позицию zscore,DOWN, цена={l_price}, в {l_time}')
                    # stop = modul.find_stop_loss(df, 'DOWN')
                    open_pair_position(coin1, coin2, "DOWN", pos_size, lookback, stop,
                                       True, strategy, go_down, go_up)
        elif strategy == 'price_range':
            if l_price < go_up:
                print(f'{coin1}/{coin2}-открываем позицию price_range,UP, цена={l_price}, в {l_time}')
            elif l_price > go_down:
                print(f'{coin1}/{coin2}-открываем позицию price_range,DOWN, цена={l_price}, в {l_time}')
        elif strategy == 'bb1':
            # для ББ1 вход - на пересечении линии наружу,
            if l_price < l_bb_down:
                print(f'{coin1}/{coin2}-открываем позицию bb1,UP, цена={l_price}, в {l_time}')
                open_pair_position(coin1, coin2, "UP", pos_size, lookback, stop,
                                   True, strategy, go_down, go_up)
            elif l_price > l_bb_up:
                print(f'{coin1}/{coin2}-открываем позицию bb1,DOWN, цена={l_price}, в {l_time}')
                open_pair_position(coin1, coin2, "DOWN", pos_size, lookback, stop,
                                   True, strategy, go_down, go_up)
        elif strategy == 'bb2':
            # для ББ2 вход - на первом заходе внутрь диапазона,
            if l_price > l_bb_down and pre_high < pre_bb_down:
                print(f'{coin1}/{coin2}-открываем позицию bb2,UP, цена={l_price}, в {l_time}')
                open_pair_position(coin1, coin2, "UP", pos_size, lookback, stop,
                                   True, strategy, go_down, go_up)
            elif l_price < l_bb_up and pre_low > pre_bb_up:
                print(f'{coin1}/{coin2}-открываем позицию bb2,DOWN, цена={l_price}, в {l_time}')
                open_pair_position(coin1, coin2, "DOWN", pos_size, lookback, stop,
                                   True, strategy, go_down, go_up)


def open_pair_position(coin1, coin2, going_to, amount, lookback, stop=0.0,
                       limit=True, strategy='zscore', up_from=0.0, down_to=0.0):

    if going_to == 'UP':
        pos_side1 = 'buy'
        pos_side2 = 'sell'
    else:
        pos_side1 = 'sell'
        pos_side2 = 'buy'

    if limit:
        order_data1 = make_limit_order(coin1, amount, pos_side1, 0.0)
        order_data2 = make_limit_order(coin2, amount, pos_side2, 0.0)
        coin1_id = order_data1.iloc[0]['id']
        coin2_id = order_data2.iloc[0]['id']
        price = order_data1.iloc[0]['price']/order_data2.iloc[0]['price']
        pos_size1 = order_data1.iloc[0]['p_size']
        pos_size2 = order_data2.iloc[0]['p_size']
    else:
        # определим минимальный лот для первой и второй ноги
        min_size1 = modul.get_coin_min_size(coin1)
        dec1 = Decimal(str(min_size1)).as_tuple().exponent * (-1)

        min_size2 = modul.get_coin_min_size(coin2)
        dec2 = Decimal(str(min_size2)).as_tuple().exponent * (-1)

        coin1_df = modul.get_last_price(coin1)
        coin2_df = modul.get_last_price(coin2)
        if going_to == 'UP':
            last_price1 = coin1_df.iloc[0]['ask']
            last_price2 = coin2_df.iloc[0]['bid']
        else:
            last_price1 = coin1_df.iloc[0]['bid']
            last_price2 = coin2_df.iloc[0]['ask']

        pos_size1 = round(amount / last_price1, dec1)
        order_data1 = place_market_order(coin1, pos_size1, pos_side1)
        pos_size2 = round(amount / last_price2, dec2)
        order_data2 = place_market_order(coin2, pos_size2, pos_side2)
        coin1_id = order_data1['id']
        coin2_id = order_data2['id']
        try:
            price = order_data1['price']/order_data2['price']
        except Exception as error:
            price = last_price1/last_price2
            print(f'ошибка расчета цены сделки - {error}')

    # добавляем пару к отслеживанию
    new_row = pd.DataFrame({
        'coin1_id': [coin1_id],
        'coin2_id': [coin2_id],
        'pair':[coin1+"-"+coin2],
        'coin1': [coin1],
        'coin2': [coin2],
        'going_to': [going_to],
        'price': [round(price, 6)],
        'stop': [stop],
        'size1': [pos_size1],
        'size2': [pos_size2],
        'strategy': [strategy],
        'lookback': [lookback],
        'up_from': [up_from],
        'down_to': [down_to],
    },
        index=None)

    save_to_log(coin1_id, new_row, True)

    if use_sql_for_report:
        with connection.connect() as conn:
            new_row.to_sql(name='bin_to_close', con=conn, if_exists='append', index=False)
    else:
        filepath_close = r'.\reports\bin_to_close.csv'
        if path.exists(filepath_close):
            # добавим к имеющимся парам для отслеживания новые
            file_close_df = pd.read_csv(filepath_close, sep="\t")
            close_df = pd.concat([file_close_df, new_row], ignore_index=True)
            close_df.to_csv(filepath_close, index=False, sep="\t")


def close_pair_position(coin1_id, coin2_id, coin1, coin2, size1, size2, l_price, new_row):
    # Close positions
    close_position(coin1_id, coin1, size1, True)
    close_position(coin2_id, coin2, size2, True)
    save_close_changes(coin1_id)
    print(f'закрыли позицию по {coin1}/{coin2}, цена={l_price}, в {datetime.datetime.now()}')
    save_to_log(coin1_id, new_row, False)


def open_one_time_positions():

    # если баланс недостаточен - нет смысла смотреть дальше
    if not modul.enough_balance():
        print('недостаточно средств на балансе')
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

            l_price = get_last_spread_price(coin1, coin2)
            if strategy == 'range':
                if (going_to == 'UP' and l_price < op_price) or (going_to == 'DOWN' and l_price > op_price):
                    open_pair_position(coin1, coin2, going_to, amount,
                                       lookback, stop, True, strategy, op_price, take_price)
                    print(f'{coin1}/{coin2}-открываем разовую позицию {strategy},{going_to}, цена={l_price}')
                    # убираем пару из таблицы
                    new_df = new_df[(new_df['coin1'] != coin1) & (new_df['coin2'] != coin2)]
            else:
                # пока для других стратегий условие не смотрим, сразу открываем
                open_pair_position(coin1, coin2, going_to, amount, lookback, stop, True, strategy, op_price, take_price)
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

    # если начался новый час - обновим h1_permission
    t_hour = datetime.datetime.now().hour
    if t_hour != now_hour:
        update_h1 = True
    else:
        update_h1 = False
    ########################################
    # анализируем открытые позиции, ищем ситуации для усреднения или для закрытия
    for index in range(len(close_df)):
        # получим данные строки
        coin1_id, coin2_id, coin1, coin2, going_to, stop,  size1, size2, strategy, lookback, up_from, down_to = (
            close_df[c].to_numpy()[index]
            for c in close_df
            if c in ["coin1_id", "coin2_id", "coin1", "coin2", "going_to", "stop",
                     "size1", "size2", "strategy", "lookback", "up_from", "down_to"]
        )
        df = calc_last_data(coin1, coin2, lookback, tf_5m)
        l_price = get_last_spread_price(coin1, coin2)

        # если разрешение на уровне Н1 не получено - закрываем открытые позиции
        # TODO - переписать обращение, сделать только раз в час
        # if update_h1:
        df_1h = calc_last_data(coin1, coin2, lookback, tf_1h)
        h1_denied = h1_permission(df_1h, False)

        # подготовим остальные данные
        last_row = df.tail(1)
        l_sma = last_row.iloc[0]['sma']
        if strategy != 'range':
            if h1_denied == 'stop':
                stop = l_price
            else:
                stop = 0.0

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
        if strategy == 'range':
            if going_to == 'UP':
                if down_to < l_price or l_price < stop:
                    close_pair_position(coin1_id, coin2_id, coin1, coin2, size1, size2, l_price, new_row)
            elif going_to == 'DOWN':
                if down_to > l_price or l_price > stop:
                    close_pair_position(coin1_id, coin2_id, coin1, coin2, size1, size2, l_price, new_row)
        else:
            if h1_denied == 'stop':
                close_pair_position(coin1_id, coin2_id, coin1, coin2, size1, size2, l_price, new_row)
            elif going_to == 'UP':
                if (l_price > l_sma) | (stop != 0.0 and l_price < stop):
                    close_pair_position(coin1_id, coin2_id, coin1, coin2, size1, size2, l_price, new_row)
            elif going_to == 'DOWN':
                if (l_price < l_sma) | (stop != 0.0 and l_price > stop):
                    close_pair_position(coin1_id, coin2_id, coin1, coin2, size1, size2, l_price, new_row)


# schedule.every(15).seconds.do(check_for_close)
schedule.every(10).minutes.do(cut_tails)
schedule.every(1).minutes.do(open_one_time_positions)
while True:
    schedule.run_pending()
    check_for_open()
    check_for_close()
    time.sleep(15)

# open_pair_position('NKNUSDT', 'ONEUSDT', 'UP', 10.0, 50, stop=0.0, limit=True)

# TODO
# 1. возможно - отмечать уровни хай/лоу в диапазоне проверки коинт/стац, и их использовать как стопы
# либо как диапазон для торговли от уровней
# 2. тейк увеличить (тестируем - до противоположного уровня / до % от уровня / смотреть по Н1 пока растут(падают) свечи)
# 2.1 Может сделать трейл-стоп. отдельную процедуру. когда цена доходит до закрытия - начинать ее трейлить.
