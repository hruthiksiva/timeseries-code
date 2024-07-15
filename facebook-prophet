import pandas as pd
from prophet import Prophet
import pandas as pd
import matplotlib.pyplot as plt

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
df = data.reset_index()
df.rename(columns={'Time': 'ds', '%user': 'y'}, inplace=True)

# Initialize the model
model = Prophet()
model.fit(df[['ds', 'y']])

# Make a future dataframe
future = model.make_future_dataframe(periods=10, freq='S')
forecast = model.predict(future)

# Plot the forecast
fig = model.plot(forecast)
plt.show()

# Evaluate the model
from sklearn.metrics import mean_absolute_error

# Make predictions on the test set
test_df = df.tail(test_size)
test_forecast = model.predict(test_df[['ds']])
test_pred = test_forecast['yhat'].values

# Calculate MAE
test_actual = test_df['y'].values
test_mae = mean_absolute_error(test_actual, test_pred)
print(f"Test MAE: {test_mae}")
