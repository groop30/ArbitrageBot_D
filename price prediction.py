import datetime

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.vector_ar.var_model import VAR
import pmdarima as pm
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from prophet import Prophet
from keras.models import Sequential
from keras.layers import LSTM, Dense
from arch import arch_model
import xgboost as xgb
from sklearn.mixture import GaussianMixture
from hmmlearn.hmm import GaussianHMM

import matplotlib.pyplot as plt
import bin_utils as modul
import datetime

connection = modul.connect_to_sqlalchemy_binance()

def prepare_data(coin1, coin2, start, end):

    coin1_df = modul.get_sql_history_price(coin1, connection, start, end)
    coin2_df = modul.get_sql_history_price(coin2, connection, start, end)
    spread_df = modul.make_spread_df(coin1_df, coin2_df, True)
    spread_df['Date'] = pd.to_datetime(spread_df['startTime'])
    spread_df['Price'] = pd.to_numeric(spread_df['close'], errors='coerce')
    spread_df.drop(labels=['time', 'startTime', 'open', 'high', 'low', 'close'],
                     axis=1,
                     inplace=True)

    return spread_df


def arima_model(train, test):

    # Split data into training and testing sets
    # data.set_index('Date', inplace=True)

    train.set_index('Date', inplace=True)
    # Create and fit the ARIMA model
    model = sm.tsa.ARIMA(train['Price'], order=(2, 1, 2))  # (p, d, q) order values can be adjusted
    model_fit = model.fit()

    start = test.iloc[0]['Date']
    end = test.iloc[len(test) - 1]['Date']

    # Predict future prices
    forecast = model_fit.predict(start=start, end=end)

    forecast_df = pd.DataFrame(forecast)
    forecast_df.reset_index(inplace=True)
    forecast_df.rename(columns={"index": "Date", "predicted_mean": "arima"}, inplace=True)
    forecast_df['Date'] = pd.to_datetime(forecast_df['Date'])
    return forecast_df


def autoarima_model(train, test):
    train.set_index('Date', inplace=True)
    # Create and fit the ARIMA model
    model = pm.auto_arima(train['Price'], seasonal=False, trace=True)  # (p, d, q) order values can be adjusted
    model_fit = model.fit()

    start = test.iloc[0]['Date']
    end = test.iloc[len(test) - 1]['Date']

    # Predict future prices
    forecast = model_fit.predict(start=start, end=end)

    forecast_df = pd.DataFrame(forecast)
    forecast_df.reset_index(inplace=True)
    forecast_df.rename(columns={"index": "Date", "predicted_mean": "arima"}, inplace=True)
    forecast_df['Date'] = pd.to_datetime(forecast_df['Date'])
    return forecast_df


def garch_model(data):

    returns = data['Close'].pct_change().dropna()  # Calculate returns

    # Split data into training and testing sets
    train_size = int(len(returns) * 0.8)
    train_returns, test_returns = returns[:train_size], returns[train_size:]

    # Fit GARCH model
    model = arch_model(train_returns, vol='GARCH', p=1, q=1)
    model_fit = model.fit()

    # Predict volatility for the test set
    volatility = model_fit.conditional_volatility
    predictions = volatility[train_size:]

    return predictions


def prophet_model(data):

    # Convert the date column to datetime format and rename the columns
    data['ds'] = pd.to_datetime(data['Date'])
    data['y'] = data['Price']

    # Create a new Prophet model
    model = Prophet()

    # Fit the model to the data
    model.fit(data)

    # Create future dates for prediction
    future = model.make_future_dataframe(periods=288, freq='5min')  # Predict future prices for 1 day

    # Make predictions
    forecast = model.predict(future)

    # Print the predicted prices
    forecast = forecast.tail(288)
    # forecast.rename(columns={"ds": "Data"}, inplace=True)
    forecast['Date'] = pd.to_datetime(forecast['ds'])
    return forecast


# Create input and output sequences for the LSTM model
def create_sequences(data, sequence_length):
    X = []
    y = []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i + sequence_length])
        y.append(data[i + sequence_length])
    return np.array(X), np.array(y)


def lstm_model(train, test):
    # Convert the date column to datetime format and set it as the index
    # data['Date'] = pd.to_datetime(data['Date'])
    train.set_index('Date', inplace=True)

    # Split the data into training and testing sets
    # train_data = data[:'2022-12-31']
    # test_data = data['2023-01-01':]

    # Scale the data
    scaler = MinMaxScaler()
    scaled_train_data = scaler.fit_transform(train)
    scaled_test_data = scaler.transform(test)

    sequence_length = 10  # Number of previous time steps to consider
    X_train, y_train = create_sequences(scaled_train_data, sequence_length)
    X_test, y_test = create_sequences(scaled_test_data, sequence_length)

    # Build the LSTM model
    model = Sequential()
    model.add(LSTM(64, activation='relu', input_shape=(sequence_length, 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit(X_train, y_train, epochs=20, batch_size=32)

    # Predict future prices
    test_input = scaled_test_data[:sequence_length].reshape(1, sequence_length, 1)
    forecast = []
    for _ in range(len(test)):
        predicted_value = model.predict(test_input)[0][0]
        forecast.append(predicted_value)
        test_input = np.append(test_input[:, 1:, :], [[predicted_value]], axis=1)

    # Inverse scale the forecasted values
    forecast = scaler.inverse_transform(forecast)

    # Print the predicted prices
    return forecast


def decision_tree_method(data):

    # Split the data into features (X) and target variable (y)
    X = data.drop(['Date', 'Price'], axis=1)
    y = data['Price']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a Decision Tree regressor
    model = DecisionTreeRegressor()

    # Fit the model to the training data
    model.fit(X_train, y_train)

    # Make predictions on the testing set
    predictions = model.predict(X_test)

    # Calculate the root mean squared error (RMSE)
    rmse = mean_squared_error(y_test, predictions, squared=False)

    # Print the RMSE
    return rmse


def multi_layer_perceptron_method(data):
    # Split the data into features (X) and target variable (y)
    X = data.drop(['Date', 'Price'], axis=1)
    y = data['Price']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a Neural Network regressor
    model = MLPRegressor(hidden_layer_sizes=(100, 50), activation='relu', random_state=42)

    # Fit the model to the training data
    model.fit(X_train, y_train)

    # Make predictions on the testing set
    predictions = model.predict(X_test)

    # Calculate the root mean squared error (RMSE)
    rmse = mean_squared_error(y_test, predictions, squared=False)

    # Print the RMSE
    return rmse


def xgboost_model(data):
    X = data.drop('Price', axis=1)  # Features
    y = data['Price']
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Step 2: Model training
    xgb_model = xgb.XGBRegressor()  # Create an instance of XGBoost regressor

    # Train the model
    xgb_model.fit(X_train, y_train)

    # Step 3: Model evaluation
    # Make predictions on the test set
    y_pred = xgb_model.predict(X_test)

    # Calculate root mean squared error (RMSE)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    print("Root Mean Squared Error (RMSE):", rmse)

    # Step 4: Prediction
    # Assume you have a new set of data for which you want to predict the price

    # # Make predictions on the new data
    # new_data = [...]  # Your new data as a list or pandas DataFrame
    # predicted_price = xgb_model.predict(new_data)

    return y_pred


# #######################################################
# Изменение параметров при изменении волатильности
# ######################################################
def calculate_volatility(price_data, window=20):
    # Calculate logarithmic returns
    returns = np.log(price_data[1:] / price_data[:-1])

    # Calculate rolling standard deviation
    volatility = returns.rolling(window).std()
    volatility = volatility.dropna()
    return volatility


def calculate_threshold(data):
    # Extract the volatility data from the provided dataset
    volatility = data.values

    # Fit Gaussian Mixture Model
    gmm = GaussianMixture(n_components=2)  # Assuming two components for simplicity
    gmm.fit(volatility.reshape(-1, 1))

    # Determine the threshold based on the larger standard deviation of the two mixture populations
    threshold_gmm = max(gmm.covariances_.flatten())

    # Fit Hidden Markov Model
    hmm = GaussianHMM(n_components=2)
    hmm.fit(volatility.reshape(-1, 1))

    # Determine the threshold based on the larger standard deviation of the two switching models
    threshold_hmm = max(hmm.covars_.flatten())

    # Select the larger threshold between GMM and HMM
    threshold = max(threshold_gmm, threshold_hmm)

    return threshold


def get_new_parameters(data):

    volatility = calculate_volatility(data['Price'], window=5)
    threshold = calculate_threshold(volatility)
    print(threshold)



def main():
    # start_time = datetime.datetime.now().timestamp() - 2000 * tf_5m
    start_time = datetime.datetime(2023, 4, 1, 0, 0, 0).timestamp()
    end_time = datetime.datetime(2023, 5, 1, 0, 0, 0).timestamp()
    coin1 = 'DASHUSDT'
    coin2 = 'MASKUSDT'
    df = prepare_data(coin1, coin2, start_time, end_time)
    train_size = int(len(df) - 288)  # пробуем предсказать один день
    train_df, test_df = df[:train_size], df[train_size:]

    # arima_df = arima_model(train_df, test_df)
    # var_df = var_model(train, test)

    # train_df.reset_index(inplace=True)
    # prophet_df = prophet_model(train_df)

    # ltsm_df = lstm_model(train_df, test_df)
    # dtree_df = decision_tree_method(df)
    # mlp_df = multi_layer_perceptron_method(df)
    signal = get_new_parameters(df)
    # xgb_df = xgboost_model(df)
    # print(xgb_df)
    # res_df = pd.concat([test_df, arima_df], ignore_index=True)
    # res_df = pd.concat([res_df, prophet_df], ignore_index=True)

    # # Plot predicted volatility
    # plt.figure(figsize=(10, 6))
    # plt.plot(test_df.index, arima_df['arima'], label='Predicted Volatility')
    # plt.plot(test_df.index, test_df['Price'], label='Actual Returns')
    # plt.xlabel('Date')
    # plt.ylabel(' Returns')
    # plt.legend()
    # plt.show()


main()