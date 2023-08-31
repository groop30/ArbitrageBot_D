import datetime
import pandas as pd
from os import path
import bin_utils as modul

pd.options.mode.chained_assignment = None


# основная процедура
def main():
    opt_list = pd.DataFrame(columns=["coin1", "coin2"])
    filepath_close = r'.\reports\to_optimize.csv'
    if path.exists(filepath_close):
        opt_list = pd.read_csv(filepath_close, sep="\t")
    # обнулим файл с итогами
    # filepath_total = r'.\optimization\total_optimize.csv'
    # total_row = pd.DataFrame()
    # total_row.to_csv(filepath_total, index=False, sep="\t")
    ########################################
    # блок условий
    up_from = 2.0
    up_to = 4.0
    down_from = -2.0
    down_to = -4.0
    step = 0.5
    step_sma = 50
    sma_from = 50
    sma_to = 250
    start = datetime.datetime(2022, 11, 3, 0, 0, 0).timestamp()
    end = datetime.datetime(2022, 11, 17, 0, 0, 0).timestamp()
    # forward = True
    # size = 1000.0
    tf = 5*60

    for index in range(len(opt_list)):
        coin1 = opt_list.iloc[index]['coin1']
        coin2 = opt_list.iloc[index]['coin2']
        pair = f'{coin1}_{coin2}'
        print(f'рассчитываем {pair}')
        df_coin1 = modul.get_history_price(coin1, start, end, tf)
        df_coin2 = modul.get_history_price(coin2, start, end, tf)
        hist_df = modul.make_spread_df(df_coin1, df_coin2)
        hist_df.sort_values(
          by='startTime',
          ascending=True,
          inplace=True,
          ignore_index=True,
        )
        temp_up = up_from
        while temp_up <= up_to:
            temp_down = down_to
            while temp_down <= down_from:
                temp_sma = sma_from
                while temp_sma <= sma_to:
                    print(f'  sma={temp_sma}, Up={temp_up}, Down={temp_down}')
                    modul.calculate_historical_profit(hist_df, pair, temp_sma, temp_up, temp_down)
                    temp_sma = temp_sma + step_sma
                temp_down = temp_down + step
            temp_up = temp_up + step


main()
