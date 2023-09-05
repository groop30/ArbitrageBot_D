import requests
import key
import pandas as pd
import datetime

tf_5m = 300


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

    api_url = f'https://api.alor.ru/md/v2/Securities?limit=1000&sector={sector}&cficode=ESXXXX&exchange=MOEX&format=Simple'
    response = requests.get(api_url, headers=headers)

    if response.status_code == 200:
        # Обработайте успешный ответ здесь
        res = response.json()
        df = pd.DataFrame(res)
        if sector == 'FOND':
            df = df[df['primary_board'] == 'TQBR']
        return df
    else:
        print('Ошибка при запросе истории цен:', response.status_code)


def get_history_price(headers, asset, start_date, end_date):
    # Пример авторизованного GET-запроса
    api_url = f'https://api.alor.ru/md/v2/history?symbol={asset}&exchange=MOEX&tf={tf_5m}&from={start_date}&to={end_date}&format=Simple'
    response = requests.get(api_url, headers=headers)

    if response.status_code == 200:
        # Обработайте успешный ответ здесь
        res = response.json()
        df = pd.DataFrame(res['history'])
        df['startTime'] = df['time'].map(lambda x: datetime.datetime.fromtimestamp(x))
        return df
    else:
        print('Ошибка при запросе истории цен:', response.status_code)


def main():

    headers = autorization()
    list = fetch_securities_list(headers, "FOND")  #FORTS, FOND, CURR
    sec_list = list['symbol']
    end_date = int(datetime.datetime.now().timestamp())
    start_date = end_date - 1000*tf_5m
    for asset in sec_list:
        ass_prices = get_history_price(headers, asset, start_date, end_date)


main()