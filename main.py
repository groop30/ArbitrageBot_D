import sys
import pandas as pd
from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5 import uic, QtWidgets, QtCore, QtGui
import bin_utils as modul
import byb_utils as modul_byb
import stratigies_tester as tester
import pyqtgraph as pg
import datetime
import BinScreener as screener
import talib
# import numpy as np

# from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg

connection = modul.connect_to_sqlalchemy_binance()
connection_bybit = modul.connect_to_sqlalchemy_bybit()

tf_5m = 5 * 60
tf_1h = 60 * 60
red_pen = pg.mkPen(color=(255, 0, 0))
blue_pen = pg.mkPen(color=(0, 0, 255))
green_pen = pg.mkPen(color=(0, 128, 0))
orange_pen = pg.mkPen(color=(255, 140, 0))

# class my_canvas(FigureCanvasQTAgg):
#     def __init__(self, fig):


class CandlestickItem(pg.GraphicsObject):
    def __init__(self, data):
        pg.GraphicsObject.__init__(self)
        self.df = data
        self.generatePicture()

    def generatePicture(self):
        self.picture = QtGui.QPicture()
        p = QtGui.QPainter(self.picture)

        for i in range(len(self.df)):
            index = self.df.index[i]
            # time = self.df.loc[index]['startTime']
            open = self.df.loc[index]['open']
            high = self.df.loc[index]['high']
            low = self.df.loc[index]['low']
            close = self.df.loc[index]['close']

            if close >= open:
                p.setPen(green_pen)
                p.setBrush(pg.mkBrush(color=(0, 128, 0)))
            else:
                p.setPen(red_pen)
                p.setBrush(pg.mkBrush(color=(255, 0, 0)))

            p.drawLine(QtCore.QPointF(i, high), QtCore.QPointF(i, low))
            p.drawRect(QtCore.QRectF(i-0.25, open, 0.5, close-open))
        p.end()

    def paint(self, p, *args):
        p.drawPicture(0, 0, self.picture)

    def boundingRect(self):
        return QtCore.QRectF(self.picture.boundingRect())


class App(QWidget):
    def __init__(self):
        super().__init__()
        self.start()
        self.refresh_opened()
        self.refresh_screened()
        self.refresh_checked()
        self.set_analyse_dates()
        self.set()

    def start(self):
        self.ui = uic.loadUi('main_window.ui')
        self.ui.show()

    def set(self):
        # Tab Trading
        self.ui.ButtonShortLim.clicked.connect(lambda: self.click_short(True))
        self.ui.ButtonLongLim.clicked.connect(lambda: self.click_long(True))
        self.ui.ButtonShortMar.clicked.connect(lambda: self.click_short(False))
        self.ui.ButtonLongMar.clicked.connect(lambda: self.click_long(False))
        self.ui.ButtonCloseLim.clicked.connect(lambda: self.click_close(True))
        self.ui.ButtonCloseMar.clicked.connect(lambda: self.click_close(False))
        # self.ui.sortDefault.clicked.connect(lambda: self.refresh_opened('default'))
        self.ui.sortName.clicked.connect(lambda: self.refresh_opened('name'))
        self.ui.deletePairButton.clicked.connect(self.delete_selected_pair)
        self.ui.updateOpenButton.clicked.connect(lambda: self.refresh_opened('default'))
        self.ui.updateCheckButton.clicked.connect(self.refresh_checked)
        self.ui.ChangeActionBtn.clicked.connect(self.change_action)
        self.ui.upButton.clicked.connect(lambda: self.update_open_levels('up'))
        self.ui.downButton.clicked.connect(lambda: self.update_open_levels('down'))
        self.ui.stopButton.clicked.connect(lambda: self.update_open_levels('stop'))
        self.ui.upButtonCheck.clicked.connect(lambda: self.update_check_levels('up'))
        self.ui.downButtonCheck.clicked.connect(lambda: self.update_check_levels('down'))
        self.ui.newLookbackBtn_2.clicked.connect(self.update_closing_lookback)
        self.ui.tradingOpenPos.selectionModel().selectionChanged.connect(self.opened_pair_clicked)
        self.ui.tradingForCheck.selectionModel().selectionChanged.connect(self.for_check_pair_clicked)
        self.ui.openInTVtrading.clicked.connect(self.open_trading_in_tview)
        # Tab Screener
        self.ui.newStatsButton.clicked.connect(self.update_stats_for_check)
        self.ui.setConfirm.clicked.connect(self.add_to_check)
        self.ui.updRowScrnButton.clicked.connect(self.refresh_screened)
        self.ui.openInTV.clicked.connect(self.open_screen_in_tview)
        self.ui.tableScreen.selectionModel().selectionChanged.connect(self.selected_pair_clicked)
        self.ui.scrPlot_5m.setBackground('w')
        self.ui.scrPlot_2.setBackground('w')
        self.ui.plotWidget.setBackground('w')
        # self.ui.setLvlUp.setInputMask('00000.00000')
        # self.ui.radioButton.setChecked(True)
        self.ui.radioScreen_2.setChecked(True)
        self.ui.posSize.setText('50.0')
        # Tab Analytics
        self.ui.tableResults.selectionModel().selectionChanged.connect(self.result_pair_clicked)
        self.ui.resPlotPair.setBackground('w')
        self.ui.resPlotSingle.setBackground('w')
        self.ui.UpdateBtn.clicked.connect(self.update_results)
        self.ui.openInTVresult.clicked.connect(self.result_in_tview)
        self.ui.UpdatePlanBtn.clicked.connect(self.fill_plan)

    def click_short(self, limit):
        new_row = self.ui.tradingForCheck.currentRow()
        pair1 = self.ui.tradingForCheck.item(new_row, 0).text()
        try:
            pair = self.ui.textPairTrade.text()
            if pair != pair1:
                print("Убедитесь, что выбранная пара соответствует активной строке")
                return False
        except:
            print("Выберите пару для торговли!")
            return False
        coin1, coin2 = modul.pair_to_coins(pair)
        going_to = "DOWN"
        amount = float(self.ui.posSize.text())
        strategy = self.ui.tradingForCheck.item(new_row, 6).text()
        lookback = int(self.ui.tradingForCheck.item(new_row, 8).text())
        up = float(self.ui.tradingForCheck.item(new_row, 1).text())
        down = float(self.ui.tradingForCheck.item(new_row, 2).text())
        modul.open_pair_position(connection, coin1, coin2, going_to, amount, lookback, stop=0.0,
                                 limit=limit, strategy=strategy, up_from=up, down_to=down,
                                 use_sql_for_report=True)
        self.refresh_opened()

    def click_long(self, limit):
        new_row = self.ui.tradingForCheck.currentRow()
        pair1 = self.ui.tradingForCheck.item(new_row, 0).text()

        try:
            pair = self.ui.textPairTrade.text()
            if pair != pair1:
                print("Убедитесь, что выбранная пара соответствует активной строке")
                return False
        except:
            print("Выберите пару для торговли!")
            return False
        coin1, coin2 = modul.pair_to_coins(pair)
        going_to = "UP"
        amount = float(self.ui.posSize.text())
        up = float(self.ui.tradingForCheck.item(new_row, 1).text())
        down = float(self.ui.tradingForCheck.item(new_row, 2).text())
        strategy = self.ui.tradingForCheck.item(new_row, 6).text()
        lookback = int(self.ui.tradingForCheck.item(new_row, 8).text())
        modul.open_pair_position(connection, coin1, coin2, going_to, amount, lookback, stop=0.0,
                                 limit=limit, strategy=strategy, up_from=up, down_to=down,
                                 use_sql_for_report=True)
        self.refresh_opened()

    def click_close(self, limit):
        pair = self.ui.textPairTrade.text()
        new_row = self.ui.tradingOpenPos.currentRow()
        pair2 = self.ui.tradingOpenPos.item(new_row, 0).text()
        if pair != pair2:
            print("Проверьте пару для закрытия! Отличается от выбранной в таблице!")
            return False
        coin1_id = int(self.ui.tradingOpenPos.item(new_row, 9).text())
        coin2_id = int(self.ui.tradingOpenPos.item(new_row, 10).text())
        size1 = float(self.ui.tradingOpenPos.item(new_row, 11).text())
        size2 = float(self.ui.tradingOpenPos.item(new_row, 12).text())
        going_to = self.ui.tradingOpenPos.item(new_row, 2).text()
        exchange = self.ui.tradingOpenPos.item(new_row, 13).text()
        coin1, coin2 = modul.pair_to_coins(pair)
        _, _, l_price = modul.get_last_spread_price(coin1, coin2)
        try:
            stop = self.ui.tradingOpenPos.item(new_row, 4).text()
            stop = float(stop)
            if (going_to == 'UP' and l_price > stop) or (going_to == 'DOWN' and l_price < stop):
                # Значит закрываемся не по превышению стопа, значит стоп не передаем
                stop = 0.0
        except:
            stop = 0.0

        # создаем строку с данными
        df_row = pd.DataFrame({
            'coin1': [coin1],
            'coin2': [coin2],
            'going_to': [going_to],
            'cl_price': [round(l_price, 6)],
            'stop': [round(stop, 6)],
        },
            index=None)
        modul.close_pair_position(connection, coin1_id, coin2_id, coin1, coin2, size1, size2, l_price, df_row, limit, exchange)
        self.refresh_opened()

    def delete_selected_pair(self):
        new_row = self.ui.tradingForCheck.currentRow()
        pair = self.ui.tradingForCheck.item(new_row, 0).text()
        exchange = self.ui.tradingForCheck.item(new_row, 7).text()
        if exchange == 'Binance':
            modul.delete_row_from_sql(connection, 'bin_to_check', pair)
        else:
            modul.delete_row_from_sql(connection_bybit, 'bin_to_check', pair)
        self.refresh_checked()

    def update_open_levels(self, up_down):

        if up_down == 'up':
            level = self.ui.upLevel.text()
            self.ui.upLevel.setText('')
        elif up_down == 'down':
            level = self.ui.downLevel.text()
            self.ui.downLevel.setText('')
        else:
            level = self.ui.stopLoss.text()
            self.ui.stopLoss.setText('')

        if level != '':
            # q_text = f'{up_down} = {level}'
            new_row = self.ui.tradingOpenPos.currentRow()
            if new_row != -1:
                coin1_id = self.ui.tradingOpenPos.item(new_row, 9).text()
                modul.update_closedf(connection, coin1_id, up_down, level)
                self.refresh_opened()
            else:
                print('Сначала выберите строку')

    def update_check_levels(self, up_down):

        if up_down == 'up':
            level = self.ui.upLvlCheck.text()
            self.ui.upLvlCheck.setText('')
        else:
            level = self.ui.downLvlCheck.text()
            self.ui.downLvlCheck.setText('')

        if level != '':
            new_row = self.ui.tradingForCheck.currentRow()
            if new_row != -1:
                pair = self.ui.tradingForCheck.item(new_row, 0).text()
                modul.update_check_df(connection, pair, up_down, level)
                self.refresh_checked()
            else:
                print('Сначала выберите строку')

    def update_closing_lookback(self):
        lookback = self.ui.newLookbacktxt_2.text()
        self.ui.newLookbacktxt_2.setText('')

        if lookback != '':
            new_row = self.ui.tradingOpenPos.currentRow()
            if new_row != -1:
                coin1_id = self.ui.tradingOpenPos.item(new_row, 9).text()
                modul.update_closedf(connection, coin1_id, 'lookback', lookback)
                self.refresh_opened()
            else:
                print('Сначала выберите строку')

    def change_action(self):

        try:
            pair = self.ui.textGraphPair_2.text()
        except:
            print("Пара не выбрана!")
            return False
        new_row = self.ui.tradingForCheck.currentRow()
        pair2 = self.ui.tradingForCheck.item(new_row, 0).text()
        action = self.ui.tradingForCheck.item(new_row, 9).text()
        if pair != pair2:
            print('На графике другая пара, повторите выбор!')
            return False
        if action == 'trade':
            modul.update_check_df(connection, pair, 'action', 'waiting')
        elif action == 'signal':
            modul.update_check_df(connection, pair, 'action', 'waiting')
        else:
            modul.update_check_df(connection, pair, 'action', 'trade')
        self.refresh_checked()

    def add_to_check(self):
        new_row = self.ui.tableScreen.currentRow()
        if new_row == -1:
            print("Пара не выбрана!")
            return False

        pair = self.ui.textGraphPair.text()
        coint = float(self.ui.tableScreen.item(new_row, 1).text())
        stat = float(self.ui.tableScreen.item(new_row, 2).text())
        exchange = self.ui.tableScreen.item(new_row, 3).text()
        coin1, coin2 = modul.pair_to_coins(pair)

        strategy = self.ui.strategyList.currentText()
        if strategy == 'manual':
            up_text = self.ui.setLvlUp.text()
            up_text = up_text.replace(',', '.')
            set_up = float(up_text)
            down_text = self.ui.setLvlDown.text()
            down_text = down_text.replace(',', '.')
            set_down = float(down_text)
        else:
            set_up = 0.0
            set_down = 0.0

        try:
            set_lkb = int(self.ui.setLookback.text())
        except:
            set_lkb = 0
        try:
            set_stop = float(self.ui.setStopLoss.text())
        except:
            set_stop = 0.0
        try:
            set_take = float(self.ui.setTake.text())
        except:
            set_take = 0.0

        action = 'signal'
        if self.ui.radioButton_2.isChecked():
            action = 'trade'

        if exchange == 'Binance':
            use_connect = connection
        else:
            use_connect = connection_bybit

        # добавляем пару к отслеживанию
        df_row = pd.DataFrame({
            'pair': [pair],
            'coin1': [coin1],
            'coin2': [coin2],
            'strategy': [strategy],
            'lookback': [set_lkb],
            'up': [set_up],
            'down': [set_down],
            'stop': [set_stop],
            'take': [set_take],
            'coint': [coint],
            'statio': [stat],
            'action': [action],
        },
            index=None)
        check_table = modul.create_check_table(use_connect)
        query = check_table.select()
        with use_connect.connect() as conn:
            check_df = pd.read_sql(sql=query, con=conn)
            test_df = check_df[check_df['pair'] == pair]
            not_yet = True
            if len(test_df) > 0:
                for index in range(len(test_df)):
                    if test_df.iloc[index]['strategy'] == strategy:
                        not_yet = False
                        print(f'В таблице отслеживания уже есть пара {pair}')
            if not_yet:
                df_row.to_sql(name='bin_to_check', con=conn, if_exists='append', index=False)
                print(f'пара {pair} добавлена в отбор')

        self.refresh_checked()
        self.refresh_screened()
        self.ui.setLvlUp.setText('')
        self.ui.setLvlDown.setText('')
        # self.ui.setLookback.setText('')
        self.ui.setStopLoss.setText('')

    def selected_pair_clicked(self):

        new_row = self.ui.tableScreen.currentRow()

        cell_value = self.ui.tableScreen.item(new_row, 0).text()
        coin1, coin2 = modul.pair_to_coins(cell_value)
        exchange = self.ui.tableScreen.item(new_row, 3).text()
        lookback = 2000
        self.ui.textGraphPair.setText(cell_value)
        # Посчитаем сколько раз монеты в выбранной паре уже используются
        if exchange == 'Binance':
            check_df = modul.get_selected_pairs(connection)
        else:
            check_df = modul.get_selected_pairs(connection_bybit)
        coin1_df = check_df[(check_df['coin1'] == coin1) | (check_df['coin2'] == coin1)]
        coin2_df = check_df[(check_df['coin1'] == coin2) | (check_df['coin2'] == coin2)]
        self.ui.textCoin1.setText(f'{coin1} использован {len(coin1_df)} раз(а)')
        self.ui.textCoin2.setText(f'{coin2} использован {len(coin2_df)} раз(а)')

        df = get_pair_marketdata(cell_value, lookback, exchange)
        df_2 = df.copy()
        max_df = df['high'].max()
        min_df = df['low'].min()
        plot_item = CandlestickItem(df)

        # Выведем график ББ.
        self.ui.scrPlot_5m.clear()
        self.ui.scrPlot_5m.enableAutoRange()
        self.ui.scrPlot_5m.addItem(plot_item)
        self.ui.scrPlot_5m.setYRange(min_df, max_df)
        df['bb_up'], df['sma'], df['bb_down'] = talib.BBANDS(df.close, 1000, 1, 1, 0)
        df['sma'] = df["close"].rolling(window=1000, min_periods=1).mean()
        self.ui.scrPlot_5m.plot(df['sma'], pen='r')
        self.ui.scrPlot_5m.plot(df['bb_up'], pen='b')
        self.ui.scrPlot_5m.plot(df['bb_down'], pen='b')

        normalize = self.ui.screenNormalize.isChecked()
        if normalize:
            df_2 = modul.get_diff_pair(df_2)
            # df_2 = modul.get_normalize_pair(df_2)
            max_df = df_2['high'].max()
            min_df = df_2['low'].min()
            plot_item = CandlestickItem(df_2)
            y1 = df_2['close'].values

            # Выведем график ББ.
            self.ui.scrPlot_2.clear()
            self.ui.scrPlot_2.enableAutoRange()
            # self.ui.scrPlot_2.addItem(plot_item)
            self.ui.scrPlot_2.plot(y1, pen='r')
            self.ui.scrPlot_2.setYRange(min_df, max_df)
            df_2['bb_up'], df_2['sma'], df_2['bb_down'] = talib.BBANDS(df_2.close, 1000, 1, 1, 0)
            df_2['sma'] = df_2["close"].rolling(window=1000, min_periods=1).mean()

            self.ui.scrPlot_2.plot(df_2['sma'], pen='r')
            self.ui.scrPlot_2.plot(df_2['bb_up'], pen='b')
            self.ui.scrPlot_2.plot(df_2['bb_down'], pen='b')
        else:
            # Выведем отдельные графики выбранных монет
            df_coin1 = get_coin_marketdata(coin1, lookback, exchange, True)
            df_coin2 = get_coin_marketdata(coin2, lookback, exchange, True)

            # create the y-axis values for each dataframe
            y1 = df_coin1['close'].values
            y2 = df_coin2['close'].values

            min_value = min(y1.min(), y2.min())
            max_value = max(y1.max(), y2.max())

            self.ui.scrPlot_2.clear()
            self.ui.scrPlot_2.plot(y1, pen='r')
            self.ui.scrPlot_2.plot(y2, pen='b')

            self.ui.scrPlot_2.setYRange(min_value, max_value)

    def opened_pair_clicked(self):

        new_row = self.ui.tradingOpenPos.currentRow()
        if new_row != -1:
            cell_value = self.ui.tradingOpenPos.item(new_row, 0).text()
            direction = self.ui.tradingOpenPos.item(new_row, 2).text()
            buy_level = self.ui.tradingOpenPos.item(new_row, 1).text()
            strategy = self.ui.tradingOpenPos.item(new_row, 7).text()
            exchange = self.ui.tradingOpenPos.item(new_row, 13).text()
            strat_lkbk = int(self.ui.tradingOpenPos.item(new_row, 8).text())
            lookback = 2000

            self.ui.textGraphPair_2.setText(cell_value)
            self.ui.textPairTrade.setText(cell_value)

            df = get_pair_marketdata(cell_value, lookback, exchange)
            max_df = df['high'].max()
            min_df = df['low'].min()
            plot_item = CandlestickItem(df)
            self.ui.plotWidget.clear()
            self.ui.plotWidget.enableAutoRange()
            self.ui.plotWidget.addItem(plot_item)
            self.ui.plotWidget.addLine(x=None, y=buy_level, pen=blue_pen)
            if strategy == 'manual':
                if direction == 'UP':
                    take = self.ui.tradingOpenPos.item(new_row, 5).text()
                else:
                    take = self.ui.tradingOpenPos.item(new_row, 6).text()
                self.ui.plotWidget.addLine(x=None, y=take, pen=green_pen)

            try:
                stop = self.ui.tradingOpenPos.item(new_row, 4).text()
                self.ui.plotWidget.addLine(x=None, y=stop, pen=red_pen)
            except:
                pass
            if strategy == 'grid_1':
                bb_sigma = 1
            elif strategy == 'bb3_atr':
                bb_sigma = 3
            else:
                bb_sigma = 1
            if strategy == 'st_dev':
                df = modul.rolling_st_dev_channels(df, strat_lkbk, 1.5)
                self.ui.plotWidget.plot(df['line_center'], pen='r')
                self.ui.plotWidget.plot(df['line_up'], pen='b')
                self.ui.plotWidget.plot(df['line_down'], pen='b')

            else:
                df['bb_up'], df['sma'], df['bb_down'] = talib.BBANDS(df.close, strat_lkbk, bb_sigma, bb_sigma, 0)
                df['sma'] = df["close"].rolling(window=strat_lkbk, min_periods=1).mean()
                self.ui.plotWidget.plot(df['sma'], pen='r')
                self.ui.plotWidget.plot(df['bb_up'], pen='b')
                self.ui.plotWidget.plot(df['bb_down'], pen='b')

            self.ui.plotWidget.setYRange(min_df, max_df)

    def for_check_pair_clicked(self):

        new_row = self.ui.tradingForCheck.currentRow()
        cell_value = self.ui.tradingForCheck.item(new_row, 0).text()
        exchange = self.ui.tradingForCheck.item(new_row, 7).text()
        try:
            up_level = self.ui.tradingForCheck.item(new_row, 1).text()
            down_level = self.ui.tradingForCheck.item(new_row, 2).text()
        except:
            up_level = 0.0
            down_level = 0.0

        lookback = 2000

        self.ui.textGraphPair_2.setText(cell_value)
        self.ui.textPairTrade.setText(cell_value)

        df = get_pair_marketdata(cell_value, lookback, exchange)
        max_df = df['high'].max()
        min_df = df['low'].min()
        plot_item = CandlestickItem(df)
        self.ui.plotWidget.clear()
        self.ui.plotWidget.enableAutoRange()
        # self.ui.plotWidget.setAutoVisible()
        self.ui.plotWidget.addItem(plot_item)
        if float(up_level) != 0.0:
            self.ui.plotWidget.addLine(x=None, y=up_level, pen=orange_pen)
        if float(down_level) != 0.0:
            self.ui.plotWidget.addLine(x=None, y=down_level, pen=orange_pen)

        df['bb_up'], df['sma'], df['bb_down'] = talib.BBANDS(df.close, 1000, 1, 1, 0)
        df['sma'] = df["close"].rolling(window=1000, min_periods=1).mean()
        # df['bb_up'] = np.where(df["bb_up"].notna(), df["bb_up"], df['sma'])
        # df['bb_down'] = np.where(df["bb_down"].notna(), df["bb_down"], df['sma'])
        self.ui.plotWidget.plot(df['sma'], pen='r')
        self.ui.plotWidget.plot(df['bb_up'], pen='b')
        self.ui.plotWidget.plot(df['bb_down'], pen='b')

        self.ui.plotWidget.setYRange(min_df, max_df)

    def result_pair_clicked(self):
        new_row = self.ui.tableResults.currentRow()

        cell_value = self.ui.tableResults.item(new_row, 0).text()
        strategy = self.ui.tableResults.item(new_row, 9).text()
        try:
            lookb_k = int(self.ui.tableResults.item(new_row, 8).text())
        except:
            lookb_k = 1000
        coin1, coin2 = modul.pair_to_coins(cell_value)
        # Найдем период для вывода. Это будет 1500 свечей до открытия, и 600 свечей после закрытия.
        cl_time_str = self.ui.tableResults.item(new_row, 5).text()
        cl_time = pd.to_datetime(cl_time_str, format='%Y-%m-%d %H:%M').timestamp()
        op_time_str = self.ui.tableResults.item(new_row, 4).text()
        op_time = pd.to_datetime(op_time_str, format='%Y-%m-%d %H:%M').timestamp()
        start_time = op_time - 1500*tf_5m
        end_time = cl_time + 600*tf_5m
        window_graph = (end_time - start_time) / tf_5m
        df = get_pair_marketdata(cell_value, window_graph, 'Binance', end_time)
        if strategy == 'st_dev':
            df = modul.rolling_st_dev_channels(df, lookb_k, 1.5)
        else:
            df['bb_up'],_, df['bb_down'] = talib.BBANDS(df.close, lookb_k, 1, 1, 0)
            df['sma'] = df["close"].rolling(window=lookb_k, min_periods=1).mean()
        df = df[lookb_k:] # Уберем первую 1000, т.к. она только для расчета ББ
        df = df.reset_index(drop=True)

        max_df = df['high'].max()
        min_df = df['low'].min()

        plot_item = CandlestickItem(df)

        self.ui.textResPair.setText(cell_value)
        # Выведем график ББ.
        self.ui.resPlotPair.clear()
        self.ui.resPlotPair.enableAutoRange()
        self.ui.resPlotPair.addItem(plot_item)
        self.ui.resPlotPair.setYRange(min_df, max_df)
        if strategy == 'st_dev':
            self.ui.resPlotPair.plot(df['line_center'], pen='r')
            self.ui.resPlotPair.plot(df['line_up'], pen='b')
            self.ui.resPlotPair.plot(df['line_down'], pen='b')
        else:
            self.ui.resPlotPair.plot(df['sma'], pen='r')
            self.ui.resPlotPair.plot(df['bb_up'], pen='b')
            self.ui.resPlotPair.plot(df['bb_down'], pen='b')

        op_time_pos = df.loc[df['startTime'] >= op_time_str].index[0]
        cl_time_pos = df.loc[df['startTime'] >= cl_time_str].index[0]
        op_price = float(self.ui.tableResults.item(new_row, 6).text())
        cl_price = float(self.ui.tableResults.item(new_row, 7).text())
        line_pen = pg.mkPen(color='b', width=2, style=QtCore.Qt.DashLine)
        line = pg.PlotCurveItem([op_time_pos, cl_time_pos], [op_price, cl_price], pen=line_pen)
        self.ui.resPlotPair.addItem(line)
        # Выведем отдельные графики выбранных монет
        df_coin1 = get_coin_marketdata(coin1, window_graph, 'Binance', True, end_time)
        df_coin2 = get_coin_marketdata(coin2, window_graph, 'Binance', True, end_time)
        df_coin1 = df_coin1[lookb_k:]
        df_coin2 = df_coin2[lookb_k:]
        # create the y-axis values for each dataframe
        y1 = df_coin1['close'].values
        y2 = df_coin2['close'].values

        min_value = min(y1.min(), y2.min())
        max_value = max(y1.max(), y2.max())

        self.ui.resPlotSingle.clear()
        self.ui.resPlotSingle.plot(y1, pen='r')
        self.ui.resPlotSingle.plot(y2, pen='b')

        self.ui.resPlotSingle.setYRange(min_value, max_value)

    def set_analyse_dates(self):
        year = datetime.datetime.now().year
        month = datetime.datetime.now().month
        day = datetime.datetime.now().day
        self.ui.dateFrom.setDate(QtCore.QDate(year, month, day))
        self.ui.dateTo.setDate(QtCore.QDate(year, month, day))

    def refresh_screened(self):
        # сначала заполним таблицу сырых результатов
        if self.ui.radioScreen.isChecked():
            filepath_binance = r'.\screening\1_raw_result.csv'
            filepath_bybit = r'.\screening\1_raw_result_bybit.csv'
        elif self.ui.radioScreen_3.isChecked():
            filepath_binance = r'.\screening\bb_touch_result.csv'
            filepath_bybit = r'.\screening\3_hard_check_bybit.csv'
        elif self.ui.radioScreen_4.isChecked():
            filepath_binance = r'.\screening\3_bb3_atr.csv'
            filepath_bybit = r'.\screening\3_hard_check_bybit.csv'
        else: #radioScreen_2 by default
            filepath_binance = r'.\screening\3_hard_check.csv'
            filepath_bybit = r'.\screening\3_hard_check_bybit.csv'

        screen_bin = pd.read_csv(filepath_binance, sep="\t")
        screen_byb = pd.read_csv(filepath_bybit, sep="\t")
        check_bin = modul.get_selected_pairs(connection)
        check_byb = modul.get_selected_pairs(connection_bybit)

        self.ui.tableScreen.setRowCount(len(screen_bin)+len(screen_byb))
        self.ui.tableScreen.setHorizontalHeaderLabels(('Pair', 'Coint', 'Stat', 'Exchange'))
        row = 0

        # Сначала выведем данные по Binance
        df_new_screen = screen_bin
        byb_new_screen = screen_byb
        need_update = False
        for index in range(len(screen_bin)):
            pair = screen_bin.iloc[index]['pair']
            df = check_bin[check_bin['pair'] == pair]
            screen_byb = screen_byb[screen_byb['pair'] != pair]  # Уберем дубль, если есть.
            # else:
            self.ui.tableScreen.setItem(row, 0, QtWidgets.QTableWidgetItem(pair))
            self.ui.tableScreen.setItem(row, 1, QtWidgets.QTableWidgetItem(str(screen_bin.iloc[index]['coint'])))
            self.ui.tableScreen.setItem(row, 2, QtWidgets.QTableWidgetItem(str(screen_bin.iloc[index]['stat_pair'])))
            self.ui.tableScreen.setItem(row, 3, QtWidgets.QTableWidgetItem('Binance'))

            if len(df) > 0:
                # Значит эта пара уже есть на отслеживании
                # (теперь не убираю, т.к. одна пара модет использоваться для разных стратегий)
                # df_new_screen = df_new_screen[df_new_screen['pair'] != pair]
                # need_update = True
                self.ui.tableScreen.item(row, 0).setBackground(QtGui.QColor('#ff9c9c'))
            row += 1
        # Потом выведем данные по Bybit
        byb_new_screen = screen_byb
        for index in range(len(screen_byb)):
            pair = screen_byb.iloc[index]['pair']
            df = check_byb[check_byb['pair'] == pair]

            # else:
            self.ui.tableScreen.setItem(row, 0, QtWidgets.QTableWidgetItem(pair))
            self.ui.tableScreen.setItem(row, 1, QtWidgets.QTableWidgetItem(str(screen_byb.iloc[index]['coint'])))
            self.ui.tableScreen.setItem(row, 2, QtWidgets.QTableWidgetItem(str(screen_byb.iloc[index]['stat_pair'])))
            self.ui.tableScreen.setItem(row, 3, QtWidgets.QTableWidgetItem('Bybit'))
            if len(df) > 0:
                # Значит эта пара уже есть на отслеживании
                # byb_new_screen = byb_new_screen[byb_new_screen['pair'] != pair]
                # need_update = True
                self.ui.tableScreen.item(row, 0).setBackground(QtGui.QColor('#ff9c9c'))
            row += 1

        self.ui.tableScreen.resizeColumnToContents(0)
        self.ui.tableScreen.resizeColumnToContents(1)
        self.ui.tableScreen.resizeColumnToContents(2)
        self.ui.tableScreen.resizeColumnToContents(3)
        self.ui.tableScreen.resizeColumnToContents(4)

        if need_update:
            df_new_screen.to_csv(filepath_binance, index=False, sep="\t")
            byb_new_screen.to_csv(filepath_bybit, index=False, sep="\t")

    def refresh_opened(self, sort='name'):
        close_bin = modul.get_open_positions(connection, True)

        close_byb = modul.get_open_positions(connection_bybit, True)
        # close_bin['exchange'] = 'Binance'
        # close_byb['exchange'] = 'Bybit'
        row = 0
        close_df = pd.concat([close_bin, close_byb], ignore_index=True)
        if sort == 'name':
            close_df.sort_values(by='pair', ascending=True, inplace=True, ignore_index=True)
        self.ui.tradingOpenPos.setRowCount(len(close_df))
        for index in range(len(close_df)):
            self.ui.tradingOpenPos.setItem(row, 0, QtWidgets.QTableWidgetItem(close_df.iloc[index]['pair']))
            self.ui.tradingOpenPos.setItem(row, 1, QtWidgets.QTableWidgetItem(str(close_df.iloc[index]['price'])))
            self.ui.tradingOpenPos.setItem(row, 2, QtWidgets.QTableWidgetItem(close_df.iloc[index]['going_to']))
            self.ui.tradingOpenPos.setItem(row, 3, QtWidgets.QTableWidgetItem(str(close_df.iloc[index]['pnl'])))
            self.ui.tradingOpenPos.setItem(row, 4, QtWidgets.QTableWidgetItem(str(close_df.iloc[index]['stop'])))
            self.ui.tradingOpenPos.setItem(row, 5, QtWidgets.QTableWidgetItem(str(close_df.iloc[index]['up'])))
            self.ui.tradingOpenPos.setItem(row, 6, QtWidgets.QTableWidgetItem(str(close_df.iloc[index]['down'])))
            self.ui.tradingOpenPos.setItem(row, 7, QtWidgets.QTableWidgetItem(str(close_df.iloc[index]['strategy'])))
            self.ui.tradingOpenPos.setItem(row, 8, QtWidgets.QTableWidgetItem(str(close_df.iloc[index]['lookback'])))
            self.ui.tradingOpenPos.setItem(row, 9, QtWidgets.QTableWidgetItem(str(close_df.iloc[index]['coin1_id'])))
            self.ui.tradingOpenPos.setItem(row, 10, QtWidgets.QTableWidgetItem(str(close_df.iloc[index]['coin2_id'])))
            self.ui.tradingOpenPos.setItem(row, 11, QtWidgets.QTableWidgetItem(str(close_df.iloc[index]['size1'])))
            self.ui.tradingOpenPos.setItem(row, 12, QtWidgets.QTableWidgetItem(str(close_df.iloc[index]['size2'])))
            self.ui.tradingOpenPos.setItem(row, 13, QtWidgets.QTableWidgetItem(str(close_df.iloc[index]['exchange'])))
            row += 1
        sum_pnl = close_df['pnl'].sum()
        self.ui.textDrawdown.setText(f'Текущая просадка: {round(sum_pnl,2)}')
        self.ui.tradingOpenPos.resizeColumnToContents(0)
        self.ui.tradingOpenPos.resizeColumnToContents(1)
        self.ui.tradingOpenPos.resizeColumnToContents(2)
        self.ui.tradingOpenPos.resizeColumnToContents(3)
        self.ui.tradingOpenPos.resizeColumnToContents(4)
        self.ui.tradingOpenPos.resizeColumnToContents(5)
        self.ui.tradingOpenPos.resizeColumnToContents(6)
        self.ui.tradingOpenPos.resizeColumnToContents(7)
        self.ui.tradingOpenPos.resizeColumnToContents(8)

    def refresh_checked(self):

        check_bin = modul.get_selected_pairs(connection)
        check_byb = modul.get_selected_pairs(connection_bybit)
        check_bin['exchange'] = 'Binance'
        check_byb['exchange'] = 'Bybit'
        full_df = pd.concat([check_bin, check_byb], ignore_index=True)
        full_df.sort_values(['action', 'pair'], ascending=[True, True], inplace=True, ignore_index=True)
        # check_byb.sort_values(by='pair', ascending=True, inplace=True, ignore_index=True)
        row = 0
        self.ui.tradingForCheck.setRowCount(len(full_df))
        for index in range(len(full_df)):
            coin1 = full_df.iloc[index]['coin1']
            coin2 = full_df.iloc[index]['coin2']
            up_level = full_df.iloc[index]['up']
            down_level = full_df.iloc[index]['down']
            coint = full_df.iloc[index]['coint']
            per_dev = full_df.iloc[index]['per_dev']
            l_price = full_df.iloc[index]['l_price']
            action = full_df.iloc[index]['action']
            self.ui.tradingForCheck.setItem(row, 0, QtWidgets.QTableWidgetItem(str(coin1+'-'+coin2)))
            self.ui.tradingForCheck.setItem(row, 1, QtWidgets.QTableWidgetItem(str(up_level)))
            self.ui.tradingForCheck.setItem(row, 2, QtWidgets.QTableWidgetItem(str(down_level)))
            self.ui.tradingForCheck.setItem(row, 3, QtWidgets.QTableWidgetItem(str(coint)))
            self.ui.tradingForCheck.setItem(row, 4, QtWidgets.QTableWidgetItem(str(full_df.iloc[index]['statio'])))
            self.ui.tradingForCheck.setItem(row, 5, QtWidgets.QTableWidgetItem(str(per_dev)))
            self.ui.tradingForCheck.setItem(row, 6, QtWidgets.QTableWidgetItem(str(full_df.iloc[index]['strategy'])))
            self.ui.tradingForCheck.setItem(row, 7, QtWidgets.QTableWidgetItem(str(full_df.iloc[index]['exchange'])))
            self.ui.tradingForCheck.setItem(row, 8, QtWidgets.QTableWidgetItem(str(full_df.iloc[index]['lookback'])))
            self.ui.tradingForCheck.setItem(row, 9, QtWidgets.QTableWidgetItem(str(action)))

            if per_dev is not None:
                if per_dev > 5.0 or per_dev < -5.0:
                    self.ui.tradingForCheck.item(row, 5).setBackground(QtGui.QColor('#ff9c9c'))
                else:
                    self.ui.tradingForCheck.item(row, 5).setBackground(QtGui.QColor('#fff'))
            if coint is not None:
                if coint > 0.1:
                    self.ui.tradingForCheck.item(row, 3).setBackground(QtGui.QColor('#ff9c9c'))
                else:
                    self.ui.tradingForCheck.item(row, 3).setBackground(QtGui.QColor('#fff'))
            if action != 'trade':
                self.ui.tradingForCheck.item(row, 0).setBackground(QtGui.QColor('#C1C1C1'))
            else:
                self.ui.tradingForCheck.item(row, 0).setBackground(QtGui.QColor('#fff'))
            row += 1

        self.ui.tradingForCheck.resizeColumnToContents(0)
        self.ui.tradingForCheck.resizeColumnToContents(1)
        self.ui.tradingForCheck.resizeColumnToContents(2)
        self.ui.tradingForCheck.resizeColumnToContents(3)
        self.ui.tradingForCheck.resizeColumnToContents(4)
        self.ui.tradingForCheck.resizeColumnToContents(5)
        self.ui.tradingForCheck.resizeColumnToContents(6)
        self.ui.tradingForCheck.resizeColumnToContents(7)
        self.ui.tradingForCheck.resizeColumnToContents(8)
        self.ui.tradingForCheck.resizeColumnToContents(9)
        # self.ui.tradingForCheck.resizeColumnToContents(10)

    def update_results(self):

        # filepath = r'.\reports\bin_to_log.csv'
        # res_df = pd.read_csv(filepath, sep="\t")
        orders_table = modul.create_orders_table(connection)
        query = orders_table.select()
        with connection.connect() as conn:
            res_df = pd.read_sql(sql=query, con=conn)
        from_date = self.ui.dateFrom.date().toPyDate()
        from_datetime = datetime.datetime.combine(from_date, datetime.time.min)
        to_date = self.ui.dateTo.date().toPyDate()
        to_datetime = datetime.datetime.combine(to_date, datetime.time.max)

        # res_df = res_df[res_df['cl_time'].str.match(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{6}') == True]
        res_df['date'] = pd.to_datetime(res_df['cl_time'])
        res_df = res_df[(res_df['date'] >= from_datetime) & (res_df['date'] <= to_datetime)]
        row = 0
        res_df = res_df[res_df['cl_price'].isna() == False]
        strategy_filter = self.ui.strategyList_3.currentText()
        if strategy_filter != '---':
            res_df = res_df[res_df['strategy'] == strategy_filter]
            
        self.ui.tableResults.setRowCount(len(res_df))
        for index in range(len(res_df)):
            res_perc = res_df.iloc[index]['per_no_commis']
            self.ui.tableResults.setItem(row, 0, QtWidgets.QTableWidgetItem(res_df.iloc[index]['pair']))
            self.ui.tableResults.setItem(row, 1, QtWidgets.QTableWidgetItem(str(res_perc)))
            self.ui.tableResults.setItem(row, 2, QtWidgets.QTableWidgetItem(str(res_df.iloc[index]['plan_profit'])))
            self.ui.tableResults.setItem(row, 3, QtWidgets.QTableWidgetItem(str(res_df.iloc[index]['result'])))
            self.ui.tableResults.setItem(row, 4, QtWidgets.QTableWidgetItem(str(res_df.iloc[index]['op_time'])[:16]))
            self.ui.tableResults.setItem(row, 5, QtWidgets.QTableWidgetItem(str(res_df.iloc[index]['cl_time'])[:16]))
            self.ui.tableResults.setItem(row, 6, QtWidgets.QTableWidgetItem(str(res_df.iloc[index]['op_price'])))
            self.ui.tableResults.setItem(row, 7, QtWidgets.QTableWidgetItem(str(res_df.iloc[index]['cl_price'])))
            self.ui.tableResults.setItem(row, 8, QtWidgets.QTableWidgetItem(str(res_df.iloc[index]['lookback'])))
            self.ui.tableResults.setItem(row, 9, QtWidgets.QTableWidgetItem(str(res_df.iloc[index]['strategy'])))
            if float(res_perc) < 0:
                self.ui.tableResults.item(row, 1).setBackground(QtGui.QColor('#ff9c9c'))
            else:
                self.ui.tableResults.item(row, 1).setBackground(QtGui.QColor('#fff'))

            row +=1

        self.ui.tableResults.resizeColumnToContents(0)
        self.ui.tableResults.resizeColumnToContents(1)
        self.ui.tableResults.resizeColumnToContents(2)
        self.ui.tableResults.resizeColumnToContents(3)
        self.ui.tableResults.resizeColumnToContents(4)
        self.ui.tableResults.resizeColumnToContents(5)
        self.ui.tableResults.resizeColumnToContents(6)
        self.ui.tableResults.resizeColumnToContents(7)
        self.ui.tableResults.resizeColumnToContents(8)
        self.ui.tableResults.resizeColumnToContents(9)

        # Вывод итоговых данных
        total_pnl = round(res_df['per_no_commis'].sum(), 2)
        self.ui.total_txt.setText(f'Total PnL ={total_pnl}%, ({len(res_df)} orders)')
        neg_df = res_df[res_df['per_no_commis'] < 0.0]
        neg_total = round(neg_df['per_no_commis'].sum(), 2)
        self.ui.negative_txt.setText(f'Negative orders sum ={neg_total}%, ({len(neg_df)})')
        posi_df = res_df[res_df['per_no_commis'] >= 0.0]
        posi_total = round(posi_df['per_no_commis'].sum(), 2)
        self.ui.positive_txt.setText(f'Positive orders sum ={posi_total}%, ({len(posi_df)})')
        plan_df = res_df[res_df['plan_profit'].isna() == False]
        plan_pnl = round(plan_df['plan_profit'].sum(), 2)
        self.ui.plan_txt.setText(f'By plan ={plan_pnl}%, ({len(plan_df)})')

    def open_screen_in_tview(self):
        new_row = self.ui.tableScreen.currentRow()
        pair = self.ui.tableScreen.item(new_row, 0).text()
        exchange = self.ui.tableScreen.item(new_row, 3).text()
        # pair = self.ui.textGraphPair.text()
        open_in_tradingview(pair, exchange)

    def open_trading_in_tview(self):
        pair = self.ui.textGraphPair_2.text()
        open_in_tradingview(pair)

    def result_in_tview(self):
        pair = self.ui.textResPair.text()
        open_in_tradingview(pair)

    def update_stats_for_check(self):
        get_new_stats()
        self.refresh_checked()

    def fill_plan(self):
        from_date = self.ui.dateFrom.date().toPyDate()
        from_datetime = datetime.datetime.combine(from_date, datetime.time.min)
        to_date = self.ui.dateTo.date().toPyDate()
        to_datetime = datetime.datetime.combine(to_date, datetime.time.max)
        fill_plan_in_log(from_datetime, to_datetime)
        self.update_results()


def get_new_stats():
    end_time = datetime.datetime.now().timestamp()
    start_time = datetime.datetime.now().timestamp() - 2000 * tf_5m
    check_df = modul.get_selected_pairs(connection)
    for index in range(len(check_df)):
        pair = check_df.iloc[index]['pair']
        coin1 = check_df.iloc[index]['coin1']
        coin2 = check_df.iloc[index]['coin2']
        coint = check_df.iloc[index]['coint']
        statio = check_df.iloc[index]['statio']
        coin1_hist = modul.get_sql_history_price(coin1, connection, start_time, end_time)
        coin2_hist = modul.get_sql_history_price(coin2, connection, start_time, end_time)
        res_row = modul.get_statistics(coin1, coin2, coin1_hist, coin2_hist, False)

        coint1 = res_row.iloc[0]['coint']
        statio1 = res_row.iloc[0]['stat_pair']
        if coint != coint1:
            modul.update_check_df(connection, pair, 'coint', coint1)
        if statio != statio1:
            modul.update_check_df(connection, pair, 'statio', statio1)


def get_pair_marketdata(pair, lookback, exchange='Binance', untill=None):

    coin1, coin2 = modul.pair_to_coins(pair)
    if untill == None:
        end_time = datetime.datetime.now().timestamp()
    else:
        end_time = untill
    start_time = end_time - lookback * tf_5m - tf_5m * 50
    is_index = coin1.find("indx")
    if exchange == 'Binance':
        if is_index == -1:
            df_coin1 = modul.get_sql_history_price(coin1, connection, start_time, end_time)
        else:
            df_coin1 = modul.get_index_history(coin1, connection, start_time, end_time)
        df_coin2 = modul.get_sql_history_price(coin2, connection, start_time, end_time)
    elif exchange == 'Binance2':
        df_coin1 = modul.get_sql_history_price(coin1, connection, start_time, end_time)
        df_coin2 = modul.get_sql_history_price(coin2, connection, start_time, end_time)
    else:
        df_coin1 = modul_byb.get_sql_history_price(coin1, connection_bybit, start_time, end_time)
        df_coin2 = modul_byb.get_sql_history_price(coin2, connection_bybit, start_time, end_time)

    df = modul.make_spread_df(df_coin1, df_coin2, True, tf_5m)
    return df


def get_coin_marketdata(coin, lookback, exchange, percent=False, untill=None):
    if untill == None:
        end_time = datetime.datetime.now().timestamp()
    else:
        end_time = untill
    start_time = end_time - lookback * tf_5m - tf_5m * 50
    is_index = coin.find("indx")
    if exchange == 'Binance':
        if is_index == -1:
            df_coin = modul.get_sql_history_price(coin, connection, start_time, end_time)
        else:
            df_coin = modul.get_index_history(coin, connection, start_time, end_time)
    else:
        df_coin = modul_byb.get_sql_history_price(coin, connection_bybit, start_time, end_time)

    df_coin.sort_values(
        by='startTime',
        ascending=True,
        inplace=True,
        ignore_index=True,
    )

    if percent:
        df_coin['shift'] = df_coin.shift(periods=1)['close']
        df_coin['per'] = (df_coin['close'] - df_coin['shift']) / df_coin['shift'] * 100
        df_coin['cumulat_per'] = round(df_coin['per'].cumsum(), 3)
        df_coin.drop(["open", "high", "low", "close", "shift", "per"], axis=1, inplace=True)
        df_coin.rename({'cumulat_per': 'close'}, axis=1, inplace=True)
        df_coin = df_coin[1:]

    return df_coin


def open_in_tradingview(pair, exchange='Binance'):

    coin1, coin2 = modul.pair_to_coins(pair)
    if exchange == 'Binance':
        url = f'https://www.tradingview.com/chart/?symbol=BINANCE:{coin1}.P/BINANCE:{coin2}.P'
    else:
        url = f'https://www.tradingview.com/chart/?symbol=BYBIT:{coin1}.P/BYBIT:{coin2}.P'
    QtGui.QDesktopServices.openUrl(QtCore.QUrl(url))


def fill_plan_in_log(from_date, to_date):
    # filepath = r'.\reports\bin_to_log.csv'
    # res_df = pd.read_csv(filepath, sep="\t")
    # res_df = res_df[res_df['cl_time'].str.match(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{6}') == True]
    orders_table = modul.create_orders_table(connection)
    query = orders_table.select()
    with connection.connect() as conn:
        res_df = pd.read_sql(sql=query, con=conn)
    res_df['date'] = pd.to_datetime(res_df['cl_time'])
    slice_df = res_df[(res_df['date'] >= from_date) & (res_df['date'] <= to_date)]
    slice_df = slice_df[slice_df['plan_profit'].isna() == True]

    for index in range(len(slice_df)):
        coin1_id = slice_df.iloc[index]['coin1_id']
        commis = slice_df.iloc[index]['commis']
        pair = slice_df.iloc[index]['pair']
        date_to = pd.to_datetime(slice_df.iloc[index]['cl_time']) + datetime.timedelta(minutes=5)
        date_from = pd.to_datetime(slice_df.iloc[index]['op_time'])
        sub_slice = res_df[res_df['pair'] == pair]
        sub_slice = sub_slice[(sub_slice['date'] < date_to) & (sub_slice['date'] > date_from)]
        orders = len(sub_slice)
        if commis > 0.0:
            pass
        else:
            commis = 0.16
        row = slice_df.iloc[index]
        plan = tester.return_one_order_result(row, orders)
        if plan != 0.0:
            plan_profit = plan - commis
            modul.update_orders_df(connection, coin1_id, 'plan_profit', plan_profit)
            # ind_row = res_df[res_df['coin1_id'] == coin1_id].index
            # res_df.loc[ind_row, 'plan_profit'] = round(plan_profit, 2)

    # res_df.drop(labels=['date'], axis=1, inplace=True)
    # res_df.to_csv(filepath, index=False, sep="\t")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = App()
    # window.show()
    app.exec_()
