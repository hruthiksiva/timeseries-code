import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error

# Define the Transformer model
class TimeSeriesTransformer(nn.Module):
    def __init__(self, num_features, num_heads, num_layers, dropout=0.1):
        super(TimeSeriesTransformer, self).__init__()
        self.transformer = nn.Transformer(
            d_model=num_features,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dropout=dropout,
        )
        self.linear = nn.Linear(num_features, num_features)

    def forward(self, src, tgt):
        transformer_output = self.transformer(src, tgt)
        output = self.linear(transformer_output)
        return output

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
columns_to_predict = ['%user', '%nice', '%system', '%iowait', '%steal', '%idle']
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data[columns_to_predict])

# Split the data
train_size = int(len(data_scaled) * 0.6)
val_size = int(len(data_scaled) * 0.2)
test_size = len(data_scaled) - train_size - val_size

train_data = data_scaled[:train_size]
val_data = data_scaled[train_size:train_size + val_size]
test_data = data_scaled[train_size + val_size:]

# Convert data to PyTorch tensors
train_tensor = torch.tensor(train_data, dtype=torch.float32).unsqueeze(1)
val_tensor = torch.tensor(val_data, dtype=torch.float32).unsqueeze(1)
test_tensor = torch.tensor(test_data, dtype=torch.float32).unsqueeze(1)

# Define model, loss function, and optimizer
model = TimeSeriesTransformer(num_features=train_tensor.shape[2], num_heads=2, num_layers=2)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 50
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    output = model(train_tensor[:-1], train_tensor[1:])
    loss = criterion(output, train_tensor[1:])
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")

# Validation
model.eval()
val_output = model(val_tensor[:-1], val_tensor[1:])
val_loss = criterion(val_output, val_tensor[1:])
print(f"Validation Loss: {val_loss.item()}")

# Testing
test_output = model(test_tensor[:-1], test_tensor[1:])
test_loss = criterion(test_output, test_tensor[1:])
print(f"Test Loss: {test_loss.item()}")

# Inverse transform the predictions
test_pred = scaler.inverse_transform(test_output.detach().numpy().squeeze())
test_actual = scaler.inverse_transform(test_tensor[1:].numpy().squeeze())

# Calculate MAE
test_mae = mean_absolute_error(test_actual, test_pred)
print(f"Test MAE: {test_mae}")
