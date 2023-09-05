import requests
import key


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


def fetch_securities_list(headers):
    api_url = 'https://api.alor.ru/md/v2/Securities/MOEX'
    response = requests.get(api_url, headers=headers)

    if response.status_code == 200:
        # Обработайте успешный ответ здесь
        res = response.json()
        return res
    else:
        print('Ошибка при запросе истории цен:', response.status_code)


def get_history_price(headers):
    # Пример авторизованного GET-запроса
    api_url = 'https://api.alor.ru/md/v2/Securities/MOEX'
    response = requests.get(api_url, headers=headers)

    if response.status_code == 200:
        # Обработайте успешный ответ здесь
        res = response.json()
        return res
    else:
        print('Ошибка при запросе истории цен:', response.status_code)


def main():

    headers = autorization()
    list = fetch_securities_list(headers)
    list2 = get_history_price(headers)
    pass

main()