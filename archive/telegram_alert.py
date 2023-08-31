import pandas as pd
import websocket
import json
import bin_utils as modul
import datetime
import time

tf_5m = 5 * 60
alerts = []
last_klines = pd.DataFrame(columns=["coin_name, time, price, sma"], index=None)
connection = modul.connect_to_sqlalchemy_binance()


def on_open(ws):
    sub_msg = {"method": "SUBSCRIBE","params":["!miniTicker@arr"],"id": 1}
    ws.send(json.dumps(sub_msg))
    print("Opened connection")

def on_message(ws, message):
    data = json.loads(message)
    alert_pump(data)


def alert_pump(data):
    global last_klines
    end_time = datetime.datetime.now().timestamp()
    last_candle_time = end_time - tf_5m
    start_time = datetime.datetime.now().timestamp() - 150 * tf_5m
    for x in data:
        coin = x['s']
        price = float(x['c'])
        need_to_add = False
        if len(last_klines) > 0:
            coin_df = last_klines[last_klines["coin_name"] == coin]
            if len(coin_df) > 0:
                current_time = coin_df.iloc[0]['time'] / 1000
                if current_time > last_candle_time:
                    last_sma = coin_df.iloc[0]['sma']
                else:
                    need_to_add = True
                    last_klines = last_klines[last_klines["coin_name"] != coin]
            else:
                need_to_add = True
        else:
            need_to_add = True

        if need_to_add:
            coin_hist = modul.get_sql_history_price(coin, connection, start_time, end_time)
            last_row = coin_hist.tail(1)
            last_sma = coin_hist['close'].mean()
            new_row = pd.DataFrame({
                                    'coin_name': [coin],
                                    'time': [last_row.iloc[0]['time']],
                                    'price': [last_row.iloc[0]['close']],
                                    'sma': ["{:.4f}".format(last_sma)],
                                    },
                                    index=None)
            last_klines = pd.concat([last_klines,new_row], ignore_index=True)

        last_sma = float(last_sma)
        difference = round((price - last_sma)/last_sma*100,2)
        if difference > 7.0 or difference < -7.0 :
            if coin not in alerts:
                print(f'{coin} изменение цены на {difference}%')
                modul.send_message_to_telegram(f'{coin} изменение цены на {difference}%')
                alerts.append(coin)
        elif coin in alerts:
            if -5.0 < difference < 5.0:
                alerts.remove(coin)


url = 'wss://fstream.binance.com/ws'

ws = websocket.WebSocketApp(url,
                            on_open=on_open,
                            on_message=on_message)
while True:
    ws.run_forever()
    print(f'Было отключение {datetime.datetime.now()}')
    time.sleep(10)
