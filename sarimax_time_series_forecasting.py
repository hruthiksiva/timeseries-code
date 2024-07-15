import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Load the data
file_path = 'output.csv'  # Replace with your file path
data = pd.read_csv(file_path)

# Remove unwanted columns
data = data.drop(columns=['AM', 'CPU'])

# Convert Time column to datetime format, assuming data is from the same day
data['Time'] = pd.to_datetime(data['Time'], format='%H:%M:%S')

# Set Time as the index
data.set_index('Time', inplace=True)

# Explicitly set the frequency of the time index
data = data.asfreq('S')

# Function to check stationarity
def check_stationarity(timeseries):
    result = adfuller(timeseries)
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    for key, value in result[4].items():
        print('Critical Values:')
        print(f'   {key}, {value}')

# Differencing to make the series stationary
def make_stationary(series):
    diff_series = series.diff().dropna()
    return diff_series

# Fit SARIMA model
def fit_sarima(series, order, seasonal_order):
    model = SARIMAX(series, order=order, seasonal_order=seasonal_order)
    model_fit = model.fit(disp=False)
    return model_fit

# Predict future values
def forecast_sarima(model_fit, steps):
    forecast = model_fit.forecast(steps=steps)
    return forecast

# Plot actual vs predicted values
def plot_forecast(train, val, val_pred, test, test_pred, title):
    plt.figure(figsize=(10, 6))
    plt.plot(train, label='Train')
    plt.plot(val, label='Validation')
    plt.plot(val_pred, color='red', label='Validation Prediction')
    plt.plot(test, label='Test')
    plt.plot(test_pred, color='orange', label='Test Prediction')
    plt.title(title)
    plt.legend()
    plt.show()

# Columns to be predicted
columns_to_predict = ['%user', '%nice', '%system', '%iowait', '%steal', '%idle']

# Order of the SARIMA model (p, d, q)
order = (5, 1, 0)  # This is an example, you may need to tune this

# Seasonal order of the SARIMA model (P, D, Q, m)
seasonal_order = (1, 1, 1, 60)  # Example values, adjust `m` to your data's seasonality

# Number of steps to forecast
forecast_steps = 10

# Train-Validation-Test split ratios
train_ratio = 0.6
val_ratio = 0.2
test_ratio = 0.2

# Process each column
for column in columns_to_predict:
    print(f"\nProcessing column: {column}")
    series = data[column].dropna()
    
    # Split the data
    train_size = int(len(series) * train_ratio)
    val_size = int(len(series) * val_ratio)
    test_size = len(series) - train_size - val_size

    train, val, test = series[:train_size], series[train_size:train_size+val_size], series[train_size+val_size:]
    
    # Check stationarity
    print("Before differencing:")
    check_stationarity(train)
    
    # Make the series stationary
    stationary_train = make_stationary(train)
    
    # Check stationarity again
    print("After differencing:")
    check_stationarity(stationary_train)
    
    # Fit SARIMA model
    model_fit = fit_sarima(stationary_train, order, seasonal_order)
    
    # Forecast future values
    val_forecast = forecast_sarima(model_fit, len(val))
    test_forecast = forecast_sarima(model_fit, len(test))
    
    # Plot the results
    plot_forecast(train, val, val_forecast, test, test_forecast, f"Forecast for {column}")
    
    # Calculate and print error metrics
    val_mae = mean_absolute_error(val, val_forecast)
    test_mae = mean_absolute_error(test, test_forecast)
    print(f"Validation MAE for {column}: {val_mae}")
    print(f"Test MAE for {column}: {test_mae}")

    # Print forecasted values
    print(f"Forecasted values for {column}:\n{test_forecast}\n")
