# import cctx
import talib
import pandas as pd
import datetime

# ftx = cctx.ftx
# help(talib.BBANDS)

#print(talib.get_functions())
coin_close = pd.DataFrame({
    'coin1':['ALGO-PERP','COMP-PERP','LINA-PERP','AXS-PERP'],
    'coin2':['TRX-PERP','RUNE-PERP','MINA-PERP','1INCH-PERP'],
    'going_to':['UP','DOWN','UP','UP'] })
# coin_close.to_csv(r'.\files\tmp.csv')
# coin_close.to_excel('tmp.xls', sheet_name='Sheet1')
my_time = datetime.datetime.strptime('2022-10-07T21:35:00+00:00',"%Y-%m-%dT%H:%M:%S%z").timestamp()
# print(my_time)
