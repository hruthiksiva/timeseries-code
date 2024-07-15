import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error


# Load the data
file_path = 'output.csv'  # Replace with your file path
data = pd.read_csv(file_path)

# Remove unwanted columns
data = data.drop(columns=['AM', 'CPU'])

# Convert Time column to datetime format
data['Time'] = pd.to_datetime(data['Time'], format='%H:%M:%S')

# Set Time as the index
data.set_index('Time', inplace=True)

# Explicitly set the frequency of the time index
data = data.asfreq('S')


# Prepare the data
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data[columns_to_predict])

# Split the data
train_size = int(len(data_scaled) * 0.6)
val_size = int(len(data_scaled) * 0.2)
test_size = len(data_scaled) - train_size - val_size

train_data = data_scaled[:train_size]
val_data = data_scaled[train_size:train_size + val_size]
test_data = data_scaled[train_size + val_size:]

# Convert data to sequences
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

seq_length = 10
train_X, train_y = create_sequences(train_data, seq_length)
val_X, val_y = create_sequences(val_data, seq_length)
test_X, test_y = create_sequences(test_data, seq_length)

# Define the model
model = Sequential([
    LSTM(50, activation='relu', input_shape=(seq_length, train_X.shape[2])),
    Dense(train_X.shape[2])
])
model.compile(optimizer='adam', loss='mse')

# Train the model
model.fit(train_X, train_y, epochs=50, validation_data=(val_X, val_y), verbose=2)

# Make predictions
test_pred = model.predict(test_X)

# Inverse transform the predictions
test_pred = scaler.inverse_transform(test_pred)
test_actual = scaler.inverse_transform(test_y)

# Calculate MAE
test_mae = mean_absolute_error(test_actual, test_pred)
print(f"Test MAE: {test_mae}")
