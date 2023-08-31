#!/usr/bin/python
# -*- coding: utf-8 -*-
import io
from urllib.parse import urlencode
# from urllib.request import urlopen
from datetime import datetime
from requests import get
from fake_headers import Headers
import pandas as pd

period=3 # задаём период. Выбор из: 'tick': 1, 'min': 2, '5min': 3, '10min': 4, '15min': 5, '30min': 6, 'hour': 7, 'daily': 8, 'week': 9, 'month': 10
start = "01.11.2022" #с какой даты начинать тянуть котировки
end = "19.01.2023" #финальная дата, по которую тянуть котировки
result_df = pd.DataFrame(columns=['pair', 'coin1', 'coin2', 'corr', 'coint', 'stat1', 'stat2'], index=None)

#каждой акции Финам присвоил цифровой код:
tickers={'ABRD':82460,'AFKS':19715,'AFLT':29,'AGRO':399716,'AKRN':17564,'ALRS':81820,'AMEZ':20702,'APTK':13855,'AQUA':35238,'ARSA':19915,
		 'ASSB':16452,'AVAN':82843,'BANE':81757,'BANEP':81758,'BISVP':35243,'BLNG':21078,'BRZL':81901,
		 'BSPB':20066,'CBOM':420694,'CHGZ':81933,'CHKZ':21000,'CHMF':16136,'CHMK':21001,'CNTL':21002,'CNTLP':81575,
		 'DIOD':35363,'DVEC':19724,'DZRD':74744,'DZRDP':74745,'ELTZ':81934,'ENRU':16440,
		 'FEES':20509,'FESH':20708,'GAZA':81997,'GAZAP':81998,'GAZC':81398,
		 'GAZP':16842,'GAZS':81399,'GAZT':82115,'GCHE':20125,'GMKN':795,'GRNT':449114,'HIMCP':81940,'HYDR':20266,'IDVP':409486,
		 'IGST':81885,'IGSTP':81887,'IRAO':20516,'IRGZ':9,'IRKT':15547,'ISKJ':17137,'JNOS':15722,
		 'JNOSP':15723,'KAZT':81941,'KAZTP':81942,'KBSB':19916,'KCHE':20030,'KCHEP':20498,'KGKC':83261,
		 'KGKCP':152350,'KLSB':16329,'KMAZ':15544,'KMEZ':22525,'KMTZ':81903,'KOGK':20710,'KRKN':81891,'KRKNP':81892,
		 'KRKOP':81906,'KROT':510,'KROTP':511,'KRSB':20912,'KRSBP':20913,
		 'KTSB':16284,'KTSBP':16285,'KUBE':522,'KUZB':83165,'KZMS':17359,'KZOS':81856,'KZOSP':81857,
		 'LIFE':74584,'LKOH':8,'LNTA':385792,'LNZL':21004,'LNZLP':22094,'LPSB':16276,'LSNG':31,'LSNGP':542,'LSRG':19736,
		 'LVHK':152517,'MAGE':74562,'MAGEP':74563,'MAGN':16782,'MERF':20947,'MFGS':30,'MFGSP':51,'MFON':152516,'MGNT':17086,
		 'MGNZ':20892,'MGTS':12984,'MGTSP':12983,'MGVM':81829,'MISB':16330,'MISBP':16331,'MNFD':80390,'MOBB':82890,'MOEX':152798,
		 'MORI':81944,'MOTZ':21116,'MRKC':20235,'MRKK':20412,'MRKP':20107,'MRKS':20346,'MRKU':20402,'MRKV':20286,'MRKY':20681,
		 'MRKZ':20309,'MRSB':16359,'MSNG':6,'MSRS':16917,'MSST':152676,'MSTT':74549,'MTLR':21018,'MTLRP':80745,'MTSS':15523,
		 'MUGS':81945,'MUGSP':81946,'MVID':19737,'NAUK':81992,'NFAZ':81287,'NKHP':450432,'NKNC':20100,'NKNCP':20101,'NKSH':81947,
		 'NLMK':17046,'NMTP':19629,'NNSB':16615,'NNSBP':16616,'NPOF':81858,'NSVZ':81929,'NVTK':17370,'ODVA':20737,'OFCB':80728,
		 'OGKB':18684,'OMSH':22891,'OMZZP':15844,'OPIN':20711,'OSMP':21006,'OTCP':407627,'PAZA':81896,'PHOR':81114,'PHST':19717,
		 'PIKK':18654,'PLSM':81241,'PLZL':17123,'PMSB':16908,'PMSBP':16909,'POLY':175924,'PRFN':83121,'PRIM':17850,'PRIN':22806,
		 'PRMB':80818,'PRTK':35247,'PSBR':152320,'QIWI':181610,'RASP':17713,'RBCM':74779,'RDRB':181755,'RGSS':181934,'RKKE':20321,
		 'RLMN':152677,'RLMNP':388313,'RNAV':66644,'RODNP':66693,'ROLO':181316,'ROSB':16866,'ROSN':17273,'ROST':20637,'RSTI':20971,
		 'RSTIP':20972,'RTGZ':152397,'RTKM':7,'RTKMP':15,'RTSB':16783,'RTSBP':16784,'RUAL':414279,'RUALR':74718,'RUGR':66893,
		 'RUSI':81786,'RUSP':20712,'RZSB':16455,'SAGO':445,'SAGOP':70,'SARE':11,'SAREP':24,'SBER':3,'SBERP':23,'SELG':81360,'SELGP':82610,
		 'SELL':21166,'SIBG':436091,'SIBN':2,'SKYC':83122,'SNGS':4,'SNGSP':13,'STSB':20087,'STSBP':20088,'SVAV':16080,'SYNG':19651,
		 'SZPR':22401,'TAER':80593,'TANL':81914,'TANLP':81915,'TASB':16265,'TASBP':16266,'TATN':825,'TATNP':826,'TGKA':18382,'TGKB':17597,
		 'TGKBP':18189,'TGKD':18310,'TGKDP':18391,'TGKN':18176,'TGKO':81899,'TNSE':420644,'TORS':16797,'TORSP':16798,'TRCN':74561,
		 'TRMK':18441,'TRNFP':1012,'TTLK':18371,'TUCH':74746,'TUZA':20716,'UCSS':175781,'UKUZ':20717,'UNAC':22843,'UNKL':82493,
		 'UPRO':18584,'URFD':75124,'URKA':19623,'URKZ':82611,'USBN':81953,'UTAR':15522,'UTII':81040,'UTSY':419504,'UWGN':414560,
		 'VDSB':16352,'VGSB':16456,'VGSBP':16457,'VJGZ':81954,'VJGZP':81955,'VLHZ':17257,'VRAO':20958,'VRAOP':20959,'VRSB':16546,
		 'VRSBP':16547,'VSMO':15965,'VSYD':83251,'VSYDP':83252,'VTBR':19043,'VTGK':19632,'VTRS':82886,'VZRZ':17068,'VZRZP':17067,
		 'WTCM':19095,'WTCMP':19096,'YAKG':81917,'YKEN':81766,'YKENP':81769,'YNDX':388383,'YRSB':16342,'YRSBP':16343,'ZHIV':181674,
		 'ZILL':81918,'ZMZN':556,'ZMZNP':603,'ZVEZ':82001}

def get_url(ticker, ticker_em):
	# Делаем преобразования дат:
	start_date = datetime.strptime(start, "%d.%m.%Y").date()
	start_date_rev = datetime.strptime(start, '%d.%m.%Y').strftime('%Y%m%d')
	end_date = datetime.strptime(end, "%d.%m.%Y").date()
	end_date_rev = datetime.strptime(end, '%d.%m.%Y').strftime('%Y%m%d')

	# пользовательские переменные
	# ticker = "SBER"  # задаём тикер
	# сервер, на который стучимся
	finam_url = "http://export.finam.ru/"

	# можно не задавать. Это рынок, на котором торгуется бумага. Для акций работает с любой цифрой. Другие рынки не проверял.
	market = 0
	########
	# periods = {'tick': 1, 'min': 2, '5min': 3, '10min': 4, '15min': 5, '30min': 6, 'hour': 7, 'daily': 8, 'week': 9, 'month': 10}
	print("ticker=" + ticker + "; period=" + str(period) + "; start=" + start + "; end=" + end)

	# Все параметры упаковываем в единую структуру. Здесь есть дополнительные параметры, кроме тех, которые заданы в шапке. См. комментарии внизу:
	params = urlencode([
		('market', market),  # на каком рынке торгуется бумага
		('em', ticker_em),  # вытягиваем цифровой символ, который соответствует бумаге.
		('code', ticker),  # тикер нашей акции
		('apply', 0),  # не нашёл что это значит.
		('df', start_date.day),  # Начальная дата, номер дня (1-31)
		('mf', start_date.month - 1),  # Начальная дата, номер месяца (0-11)
		('yf', start_date.year),  # Начальная дата, год
		('from', start_date),  # Начальная дата полностью
		('dt', end_date.day),  # Конечная дата, номер дня
		('mt', end_date.month - 1),  # Конечная дата, номер месяца
		('yt', end_date.year),  # Конечная дата, год
		('to', end_date),  # Конечная дата
		('p', period),  # Таймфрейм
		('f', ticker + "_" + start_date_rev + "_" + end_date_rev),  # Имя сформированного файла
		('e', ".csv"),  # Расширение сформированного файла
		('cn', ticker),  # ещё раз тикер акции
		('dtf', 1),
		# В каком формате брать даты. Выбор из 5 возможных. См. страницу https://www.finam.ru/profile/moex-akcii/sberbank/export/
		('tmf', 1),  # В каком формате брать время. Выбор из 4 возможных.
		('MSOR', 0),  # Время свечи (0 - open; 1 - close)
		('mstime', "on"),  # Московское время
		('mstimever', 1),  # Коррекция часового пояса
		('sep', 1),  # Разделитель полей	(1 - запятая, 2 - точка, 3 - точка с запятой, 4 - табуляция, 5 - пробел)
		('sep2', 1),  # Разделитель разрядов
		('datf', 1),  # Формат записи в файл. Выбор из 6 возможных.
		('at', 1)])  # Нужны ли заголовки столбцов
	url = finam_url + ticker + "_" + start_date_rev + "_" + end_date_rev + ".csv?" + params  # урл составлен!

	return url


def fetch_marketdata():

	header = Headers(
		browser="chrome",  # Generate only Chrome UA
		os="win",  # Generate ony Windows platform
		headers=True  # generate misc headers
	)

	for key in tickers:
		ticker = key
		ticker_em = tickers[key]
		filepath_log = fr'.\files\{ticker}.csv'
		url = get_url(ticker, ticker_em)
		# print(f'Получаем данные по {ticker}')
		response = get(url, headers=header.generate())
		if len(response.text) > 0:
			sss = response.content
			df = pd.read_csv(io.StringIO(sss.decode('utf-8')))
			df.to_csv(filepath_log, index=False, sep="\t")
		else:
			print(f'Ticker {ticker} not exist anymore! Remove it from the list.')

def get_screening_result(lookback):

	global result_df
	# global exception_list

    # получим список всех монет
	all_tickers = tickers
	all_tickers_2 = tickers
	start_time = start
	end_time = end

	# TODO - дописать код ниже.
	for ticker in all_tickers:
		# каждую монету сравним со всеми остальными
        coin1_hist = modul.get_sql_history_price(ticker, start_time, end_time)
		if len(coin1_hist) > 0:
			print("сравниваем акцию " + ticker)
			for ticker2 in all_tickers_2:

				if ticker2 != ticker:
					# coin2_hist = modul.get_history_price(future2, start_time, end_time, tf_5m)
					coin2_hist = modul.get_sql_history_price(ticker2, start_time, end_time)
					if len(coin2_hist) > 0:
						new_row = get_statistics(ticker, ticker2, coin1_hist, coin2_hist, True)
						if new_row is not None:
							result_df = pd.concat([result_df, new_row], ignore_index=True)
							print('добавлены данные по паре ' + ticker + '/' + ticker2)
							result_df.to_csv(r'.\screening\1_raw_result.csv', index=False, sep="\t")
					else:
						print("по монете "+ticker2+" данные не получены")
		else:
			print("по монете " + ticker + " данные не получены")

		all_tickers_2.remove(ticker)





fetch_marketdata()



# local_file = open('quotes.txt', "w") #задаём файл, в который запишем котировки.

# txt=urlopen(url).readlines() #здесь лежит огромный массив данных, прилетевший с Финама.

# for line in txt: #записываем свечи строку за строкой.
# 	local_file.write(line.strip().decode( "utf-8" )+'\n')
# local_file.close()
