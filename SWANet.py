import numpy as np
import pandas as pd
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Dropout, Conv1D, MaxPooling1D, Flatten, Concatenate
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from pywt import cwt, ContinuousWavelet, scale2frequency
from scipy.stats import zscore
import datetime
import bin_utils as modul


connection = modul.connect_to_sqlalchemy_binance()

# Define the SWANet model
def create_swanet_model(input_shape, num_classes):
    # Define the CWT branch
    cwt_input = Input(shape=input_shape)
    cwt_conv = Conv1D(filters=16, kernel_size=3, activation='relu')(cwt_input)
    cwt_pool = MaxPooling1D(pool_size=2)(cwt_conv)
    cwt_flatten = Flatten()(cwt_pool)

    # Define the LSTM branch
    lstm_input = Input(shape=input_shape)
    lstm_lstm = LSTM(units=64, return_sequences=True)(lstm_input)
    lstm_flatten = Flatten()(lstm_lstm)

    # Concatenate the CWT and LSTM branches
    concat = Concatenate()([cwt_flatten, lstm_flatten])
    dense = Dense(64, activation='relu')(concat)
    output = Dense(num_classes, activation='softmax')(dense)

    # Create the SWANet model
    model = Model(inputs=[cwt_input, lstm_input], outputs=output)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# Load and preprocess the data
start_time = datetime.datetime(2023, 4, 8, 0, 0, 0).timestamp()
end_time = datetime.datetime(2023, 5, 15, 0, 0, 0).timestamp()
coin1 = 'AAVEUSDT'
coin2 = 'KLAYUSDT'
coin1_df = modul.get_sql_history_price(coin1, connection, start_time, end_time)
coin2_df = modul.get_sql_history_price(coin2, connection, start_time, end_time)
# Extract the series for stock i and stock j
series_i = coin1_df['close'].values
series_j = coin2_df['close'].values

# Normalize the series using z-score normalization
series_i = zscore(series_i)
series_j = zscore(series_j)

# Apply continuous wavelet transform (CWT) to the series
wavelet = ContinuousWavelet('gaus1')
scales = np.arange(1, 128)  # Adjust the range of scales as needed
cwt_i, _ = cwt(series_i, scales, wavelet)
cwt_j, _ = cwt(series_j, scales, wavelet)

# Define the target labels (0: no break, 1: break)
labels = np.array([0] * len(series_i))
# Ensure the same number of samples in input data and labels
min_samples = min(len(cwt_i), len(cwt_j), len(labels))
# cwt_i = cwt_i[:min_samples]
# cwt_j = cwt_j[:min_samples]
# Reshape the input data to have an additional dimension
cwt_i = cwt_i[..., np.newaxis]
cwt_j = cwt_j[..., np.newaxis]

labels = labels[:min_samples]

# Split the data into training and testing sets
X_train_cwt, X_test_cwt, X_train_lstm, X_test_lstm, y_train, y_test = train_test_split(
    cwt_i, cwt_j, labels, test_size=0.2, random_state=42
)

# Convert the labels to categorical format
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# # Reshape the input data for LSTM
# input_shape = (X_train_lstm.shape[1], 1)
#
# # Create the SWANet model
# model = create_swanet_model(input_shape, num_classes=2)
#
# # Train the model
# early_stopping = EarlyStopping(patience=5, monitor='val_loss')
# model.fit([X_train_cwt, X_train_lstm], y_train, validation_data=([X_test_cwt, X_test_lstm], y_test),
#           epochs=100, batch_size=32, callbacks=[early_stopping])

# Create and train the SWANet model
input_shape = X_train_cwt.shape[1:]
num_classes = y_train.shape[1]
model = create_swanet_model(input_shape, num_classes)
early_stopping = EarlyStopping(patience=10, restore_best_weights=True)
model.fit([X_train_cwt, X_train_lstm], y_train,
          validation_data=([X_test_cwt, X_test_lstm], y_test),
          epochs=100, batch_size=64, callbacks=[early_stopping])

predicted_prices = model.predict([X_test_cwt, X_test_lstm])
# Evaluate the model
_, accuracy = model.evaluate([X_test_cwt, X_test_lstm], y_test)
print('Accuracy:', accuracy)
