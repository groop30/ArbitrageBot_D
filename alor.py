import requests
import key
import pandas as pd
import datetime
import sqlalchemy as sql
import bin_utils as modul

tf_5m = 300


# ######################################################################
# Процедуры для работы с SQL
#
# ######################################################################
def connect_to_sqlalchemy_moex():
    engine = sql.create_engine(f'mysql+mysqlconnector://root:{key.mysqlroot}@127.0.0.1:3306/moex', echo=False)
    return engine


def create_olhcv_table(coin, connect):
    meta = sql.MetaData()

    olhc_table = sql.Table(
        coin, meta,
        sql.Column('time', sql.BIGINT, primary_key=True),
        sql.Column('startTime', sql.DateTime),
        sql.Column('open', sql.Float),
        sql.Column('high', sql.Float),
        sql.Column('low', sql.Float),
        sql.Column('close', sql.Float),
        sql.Column('volume', sql.Float),
    )
    meta.create_all(connect)
    return olhc_table


# ######################################################################
# Процедуры запроса, хранения, записи истории цен
#
# ######################################################################
def add_candels_to_database(asset, df, start_date, end_date, connection, headers):
    """
        :param coin: текстовое наименование монеты
        :param df: датафрейм с историческими ценами
        :param start: момент старта в формате timestamp
        :param lookforward: количество свечей до конца нужного периода (в указанном таймфрейме (не в секундах!))
        :param connection: соединение с базой
        :return:
        """

    if len(df) == 0:
        # если массив пустой - значит первичное наполнение
        df = fetch_alor_history_price(headers, asset, start_date, end_date)
        with connection.connect() as conn:
            try:
                df.to_sql(name=asset.lower(), con=conn, if_exists='append', index=False)
            except ValueError:
                pass
                # print(f'Запись в базу не получилась - {coin}')  #{error}
    else:
        # наиболее вероятно - что нет последних данных
        last_candle_time = df.tail(1).iloc[0]['time']
        first_candle_time = df.head(1).iloc[0]['time']
        lookforward = int((end_date - last_candle_time)/tf_5m)
        if lookforward < 0:
            lookforward = 0
        lookbackward = int((first_candle_time - start_date) / tf_5m)
        if lookforward > 0:
            t_start = last_candle_time+tf_5m
            df_temp = fetch_alor_history_price(headers, asset, t_start, end_date)
            df_temp = modul.remove_dublicates(df_temp, df)
            with connection.connect() as conn:
                try:
                    df_temp.to_sql(name=asset.lower(), con=conn, if_exists='append', index=False)
                except:
                    pass
                    # print(f'Запись в базу не получилась - {coin}')  #{error}
            df = pd.concat([df, df_temp], ignore_index=True)
            df = modul.prepare_dataframe(df=df, timestamp_field="startTime", asc=True)
        if lookbackward > 0:
            df_temp = fetch_alor_history_price(headers, asset, start_date, first_candle_time)
            df_temp = modul.remove_dublicates(df_temp, df)
            with connection.connect() as conn:
                try:
                    df_temp.to_sql(name=asset.lower(), con=conn, if_exists='append', index=False)
                except ValueError:
                    pass
                    # print(f'Запись в базу не получилась - {coin}')  #{error}
            df = pd.concat([df, df_temp], ignore_index=True)
            df = modul.prepare_dataframe(df=df, timestamp_field="startTime", asc=True)
        # потом проверяем на пропуски
        intervals = modul.get_fetch_intervals(df=df, date_column_label="time", timeframe=tf_5m)
        # Заполняем пропуски
        if lookforward == 0:
            lookforward = 1000
        for period in intervals:
            s_start = period[0]
            s_end = period[1]
            # while s_start <= period[1]:
            df_temp = fetch_alor_history_price(headers, asset, s_start, s_end)
            df_temp = modul.remove_dublicates(df_temp, df)
            with connection.connect() as conn:
                try:
                    df_temp.to_sql(name=asset.lower(), con=conn, if_exists='append', index=False)
                except ValueError:
                    pass
                    # print(f'Запись в базу не получилась - {coin}')  #{error}
            df = pd.concat([df, df_temp], ignore_index=True)
            df = modul.prepare_dataframe(df=df, timestamp_field="startTime", asc=False)
            # s_start = s_start + 1000*tf_5m
    return df


def get_sql_history_price(asset, connection, start, end, headers):
    """
    :param asset:
    :param start:  момент с которого нужны данные в формате timestamp
    :param end:  момент окончания в формате timestamp
    :param connection: соединение с базой sql
    :return:
    """

    # 1. выбираем данные за нужный период из базы
    coin_table = create_olhcv_table(asset, connection)
    query = coin_table.select().where(coin_table.columns.time >= start, coin_table.columns.time <= end)
    with connection.connect() as conn:
        history_df = pd.read_sql(sql=query, con=conn)
    # 2. смотрим, полные ли они, нужна ли дозагрузка
    lookforward = int((end - start) / tf_5m)
    if len(history_df) < lookforward:
        # 3. если да - дозагружаем
        history_df = add_candels_to_database(asset, history_df, start, end, connection, headers)
    # 4. возвращаем результат
    return history_df


# ######################################################################
# Процедуры взаимодействия с Alor API
#
# ######################################################################
def autorization():
    # Установите URL-адрес и Refresh Token
    refresh_token = key.AlorToken
    url = 'https://oauth.alor.ru/refresh?token=' + refresh_token

    response = requests.post(url)

    if response.status_code == 200:
        # Получите новый JWT-токен из ответа
        new_access_token = response.json().get('AccessToken')
        # Теперь вы можете использовать новый токен для авторизованных запросов
        headers = {'Authorization': 'Bearer ' + new_access_token}

    else:
        headers = ''

    return headers


def fetch_securities_list(headers, sector):

    if sector == 'FORTS':
        list_one = ['CRU3','SIU3','EuU3', 'NGU3', 'BRU3', 'EDU3', 'GDU3', 'RIU3', 'MMU3', 'SVU3']  # временное решение. надо разобраться как отбирать последние контракты
        full_df = pd.DataFrame()
        for asset in list_one:
            api_url = f'https://api.alor.ru/md/v2/Securities?query={asset}&limit=1000&sector={sector}&exchange=MOEX&format=Simple'
            response = requests.get(api_url, headers=headers)
            if response.status_code == 200:
                res = response.json()
                df = pd.DataFrame(res)
                full_df = pd.concat([full_df, df.head(1)], ignore_index=True)

        return full_df
    elif sector == 'FOND':
        api_url = f'https://api.alor.ru/md/v2/Securities?limit=1000&sector={sector}&cficode=ESXXXX&exchange=MOEX&format=Simple'
        response = requests.get(api_url, headers=headers)
        if response.status_code == 200:
            res = response.json()
            df = pd.DataFrame(res)
            df = df[df['primary_board'] == 'TQBR']
            return df
        else:
            print('Ошибка при запросе истории цен:', response.status_code)
    # else:


def fetch_alor_history_price(headers, asset, start_date, end_date):
    # Пример авторизованного GET-запроса
    api_url = f'https://api.alor.ru/md/v2/history?symbol={asset}&exchange=MOEX&tf={tf_5m}&from={int(start_date)}' \
              f'&to={int(end_date)}&format=Simple'
    response = requests.get(api_url, headers=headers)

    if response.status_code == 200:
        # Обработайте успешный ответ здесь
        res = response.json()
        df = pd.DataFrame(res['history'])
        if len(df) > 0:
            df['startTime'] = df['time'].map(lambda x: datetime.datetime.fromtimestamp(x))
        return df
    else:
        print('Ошибка при запросе истории цен:', response.status_code)
        return pd.DataFrame()


# def main():
#
#     headers = autorization()
#     list = fetch_securities_list(headers, "FOND")  #FORTS, FOND, CURR
#     sec_list = list['symbol']
#     end_date = int(datetime.datetime.now().timestamp())
#     start_date = datetime.datetime(2023, 8, 1, 0, 0, 0).timestamp()
#     # start_date = end_date - 1000*tf_5m
#     # for asset in sec_list:
#         # ass_prices = get_history_price(headers, asset, start_date, end_date)
#
#
# main()