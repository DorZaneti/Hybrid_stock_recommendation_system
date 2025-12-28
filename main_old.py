#import libreries
import math
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics.pairwise import pairwise_distances


if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

#Download stocks data
def download_stock_data(tickers, start_date, end_date):
    all_data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True, group_by='ticker')
    data_dict ={ticker:all_data[ticker] for ticker in tickers}
    return data_dict

tickers = [
     'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NFLX', 'NVDA', 'BABA', 'INTC',
     'AMD', 'V', 'JPM', 'DIS', 'PYPL', 'CSCO', 'PEP', 'KO', 'NKE', 'PFE',
     'ADBE', 'MRNA', 'BA', 'WMT', 'XOM', 'COST', 'ORCL', 'CRM', 'SBUX',
     'SPOT', 'SQ', 'ZM', 'SNOW', 'SHOP', 'RBLX', 'T', 'VZ', 'F', 'GM',
     'MCD', 'UPS', 'IBM', 'WFC', 'GS', 'CAT', 'TXN', 'LMT', 'GE', 'PLTR',
     'ABBV', 'MO', 'CVX', 'QCOM', 'NIO', 'JD', 'NVS', 'TM', 'RIO', 'UL',
     'BMY', 'COP', 'SCHW', 'DE', 'MS', 'TMO', 'NOC', 'DD', 'FDX', 'BKNG',
     'ISRG', 'LRCX', 'MU', 'GILD', 'AMAT', 'TGT', 'EL', 'ADI', 'C', 'PM',
     'VLO', 'LOW', 'SPGI', 'MRK', 'KMB', 'DOW', 'PGR', 'CL', 'OXY', 'NEE',
     'EXC', 'SRE', 'USB', 'PSX', 'SO', 'AEP', 'PNC', 'KHC', 'MDLZ', 'BSX',
]
data = download_stock_data(tickers,'2020-01-01','2025-12-31')
print(data['AAPL'].head())

#SIMILARITY
def calculate_rsi(df, window=14):
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_momentum(df, window=10):
    return df['Close'].diff(window)

def calculate_bollinger_bands(df, window=20):
    rolling_mean = df['Close'].rolling(window).mean()
    rolling_std = df['Close'].rolling(window).std()
    upper_band = rolling_mean + (rolling_std * 2)
    lower_band = rolling_mean - (rolling_std * 2)
    return upper_band, lower_band

def calculate_features(df):
    df['RSI'] = calculate_rsi(df)
    df['Momentum'] = calculate_momentum(df)
    df['Moving_Average'] = df['Close'].rolling(window=20).mean()
    df['Bollinger_Upper'], df['Bollinger_Lower'] = calculate_bollinger_bands(df)
    return df

for ticker in tickers:
    data[ticker] = calculate_features(data[ticker])

def find_similar_stocks(data, base_ticker):
    features = ['RSI', 'Momentum', 'Moving_Average', 'Bollinger_Upper', 'Bollinger_Lower', 'Close']

    # Get base_features and drop NaNs. If empty, we can't proceed.
    base_features = data[base_ticker][features].dropna()
    if base_features.empty:
        print(f"Warning: Base ticker '{base_ticker}' has no valid features after dropping NaNs. Returning empty list.")
        return []

    similarities = {}
    for ticker in data:
        if ticker != base_ticker:
            other_features = data[ticker][features].dropna()
            # Only calculate distance if other_features is not empty
            if not other_features.empty:
                distance = pairwise_distances(base_features, other_features, metric='euclidean').mean()
                similarities[ticker] = distance
            else:
                print(f"Warning: Ticker '{ticker}' has no valid features after dropping NaNs. Skipping similarity calculation for this ticker.")

    similar_stocks = sorted(similarities, key=similarities.get)[:10]
    return similar_stocks

similar_stocks = find_similar_stocks(data, 'AMZN')
print("Similar stocks:", similar_stocks)


#Download similar stocks data

processed_stock_data = {}
for ticker in similar_stocks:
    if ticker in data: # Check if the ticker exists in the previously downloaded 'data'
        # Extract the 'Close' prices from the already existing 'data' DataFrame for the given ticker.
        # Apply the specified date range.
        # Adjusted to access the column by ticker name, as 'Close' column seems to be missing.
        # Assuming the single price column is named after the ticker itself.
        if ticker in data[ticker].columns:
            close_prices = data[ticker][ticker].loc['2020-01-01':'2023-01-01']
        else:
            # Fallback if the column isn't named after the ticker, try 'Close'
            # This path is less likely given the KeyError, but good for robustness
            if 'Close' in data[ticker].columns:
                close_prices = data[ticker]['Close'].loc['2020-01-01':'2023-01-01']
            else:
                print(f"Warning: Neither '{ticker}' nor 'Close' column found for {ticker}. Skipping.")
                close_prices = pd.Series([], dtype='float64') # Empty Series

        if not close_prices.empty:
            # Rename the series to 'Close' for consistency in the final DataFrame structure if needed downstream
            processed_stock_data[ticker] = close_prices.rename(ticker) # Renaming the series to the ticker name to match the DataFrame columns
        else:
            print(f"Warning: 'Close' data for {ticker} is empty after applying date range to existing data. Skipping.")
    else:
        print(f"Warning: Ticker '{ticker}' from similar_stocks not found in the initial 'data' dictionary. Skipping.")

filtered_data_dict = {ticker: series for ticker, series in processed_stock_data.items() if not series.empty}

if not filtered_data_dict:
    print("No valid stock data available to create DataFrame for selected similar tickers. Creating an empty DataFrame.")
    df = pd.DataFrame()
    # Update the 'tickers' list to be empty if df is empty to prevent errors in subsequent cells.
    tickers = []
else:
    # Align indices to create a clean DataFrame
    common_index = None
    for series in filtered_data_dict.values():
        if common_index is None:
            common_index = series.index
        else:
            common_index = common_index.intersection(series.index)

    if common_index.empty:
        print("No common dates found across all selected similar tickers. Creating an empty DataFrame.")
        df = pd.DataFrame()
        tickers = []
    else:
        aligned_data_dict = {ticker: series.reindex(common_index) for ticker, series in filtered_data_dict.items()}
        df = pd.DataFrame(aligned_data_dict)
        # Update the global 'tickers' list to match the columns of the new 'df'.
        tickers = list(df.columns)

print(df.head())


#LSTM
#Preprocess DATA
#scaling the data and handling missing values (between 0 to 1)

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df.dropna())

print(scaled_data)

#creating sequences of data (the 3rd dimension required by LSTM)

def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:i+seq_length]
        y = data[i+seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)
seq_length = 8
X, y = create_sequences(scaled_data, seq_length)

#splitting the data into train and test sets

X_train, y_train = torch.tensor(X[:-5]), torch.tensor(y[:-5])
X_test, y_test = torch.tensor(X[-5:]), torch.tensor(y[-5:])

#splitting the data into 2 training sets: one for the LSTM model with 4 unfrozen layers and one for the LSTM model with first 3 frozen layers

X_train_1, y_train_1 = torch.tensor(X[:-20]), torch.tensor(y[:-20])
X_train_2, y_train_2 = torch.tensor(X[-20:-5]), torch.tensor(y[-20:-5])

#Define LSTM model
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm1 = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True, dropout=0.4)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, num_layers=1, batch_first=True, dropout=0.4)
        self.lstm3 = nn.LSTM(hidden_size, hidden_size, num_layers=1, batch_first=True, dropout=0.4)
        self.lstm4 = nn.LSTM(hidden_size, hidden_size, num_layers=1, batch_first=True, dropout=0.4)
#        self.lstm5 = nn.LSTM(hidden_size, hidden_size, num_layers=1, batch_first=True, dropout=0.4)
#        self.lstm6 = nn.LSTM(hidden_size, hidden_size, num_layers=1, batch_first=True, dropout=0.4)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm1(x, (h0, c0))
        out, _ = self.lstm2(out, (h0, c0))
        out, _ = self.lstm3(out, (h0, c0))
        out, _ = self.lstm4(out, (h0, c0))
#        out, _ = self.lstm5(out, (h0, c0))
#        out, _ = self.lstm6(out, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

#LSTM model with 4 hidden layers (each layer was defined separetely in order to eventually be frozen later)


#Build LSTM model with len(tickers) features
model = LSTM(input_size=len(tickers), hidden_size=100, num_layers=1, output_size=len(tickers)).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

#train LSTM model on data_length - 5(frist training set)
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    outputs = model(X_train.float().to(device))
    optimizer.zero_grad()
    loss = criterion(outputs, y_train.float().to(device))
    loss.backward()
    optimizer.step()
    scheduler.step()
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

#Evaluate the modle on the test set
# Set the model to evaluation mode
model.eval()

# Make predictions on the test set
outputs = model(X_test.float().to(device))

# Calculate the loss function (MSE) on the test set
test_loss = criterion(outputs, y_test.float().to(device))

# Print the test loss
print(f'Test Loss: {test_loss.item():.4f}')

# If you want to calculate the RMSE, you can take the square root of the MSE
print(f'Test RMSE: {math.sqrt(test_loss.item()):.4f}')

#Predict future prices
scaled_predicted_prices = outputs.cpu().detach().numpy()
predicted_prices_0 = scaler.inverse_transform(scaled_predicted_prices)
print(predicted_prices_0)

#Build a second LSTM model with len(tickers) features
model = LSTM(input_size=len(tickers), hidden_size=100, num_layers=1, output_size=len(tickers)).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

#Train LSTM model on train_1 data(till data_length - 20 trading days)
for epoch in range(num_epochs):
    model.train()
    outputs = model(X_train_1.float().to(device))
    optimizer.zero_grad()
    loss = criterion(outputs, y_train_1.float().to(device))
    loss.backward()
    optimizer.step()
    scheduler.step()
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

#Freeze the first three LSTM layers
for param in model.lstm1.parameters():
    param.requires_grad = False
for param in model.lstm2.parameters():
    param.requires_grad = False
for param in model.lstm3.parameters():
    param.requires_grad = False

#train the LSTM model with four frozen layers on train_2 data(from data lenght 20 to data lenght 5)
for epoch in range(num_epochs):
    model.train()
    outputs = model(X_train_2.float().to(device))
    optimizer.zero_grad()
    loss = criterion(outputs, y_train_2.float().to(device))
    loss.backward()
    optimizer.step()
    scheduler.step()
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

#Evaluate the model on the test set
# Set the model to evaluation mode
model.eval()

# Make predictions on the test set
outputs = model(X_test.float().to(device))

# Calculate the loss function (MSE) on the test set
test_loss = criterion(outputs, y_test.float().to(device))

# Print the test loss
print(f'Test Loss: {test_loss.item():.4f}')

# If you want to calculate the RMSE, you can take the square root of the MSE
print(f'Test RMSE: {math.sqrt(test_loss.item()):.4f}')

#predict future prices
scaled_predicted_prices = outputs.cpu().detach().numpy()
predicted_prices_0f = scaler.inverse_transform(scaled_predicted_prices)
print(predicted_prices_0f)

#Build LSTM model for only one ticker at time(model with only one feature)
model_1 = LSTM(input_size=1, hidden_size=100, num_layers=1, output_size=1).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model_1.parameters(), lr=0.002)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

#Train the Model one data length 5- first traning set
num_epochs = 10
for epoch in range(num_epochs):
  for i in range(int(len(tickers))):
        # Slice data for current ticker
        X_train_i = X_train[:, :,i:i+1]
        y_train_i = y_train[:,i:i+1]
        model_1.train()
        outputs = model_1(X_train_i.float().to(device))
        optimizer.zero_grad()
        loss = criterion(outputs, y_train_i.float().to(device))
        loss.backward()
        optimizer.step()
        scheduler.step()
  if (epoch+1) % 2 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

#evaluate the model
my_outputs_1 = torch.zeros(5,10).to(device)
for i in range(len(tickers)):
    # Calculate my_outputs for current ticker
    model_1.eval()
    my_local_outputs = model_1(X_test[:, :, i:i+1].float().to(device))
    # Process my_outputs for current ticker
    my_outputs_1[:, i:i+1] = my_local_outputs
test_loss = criterion(my_outputs_1, y_test.float().to(device))
print(f'Test Loss: {test_loss.item():.4f}')
print(f'Test RMSE: {math.sqrt(test_loss.item()):.4f}')
print(my_outputs_1)

#predict future prices
scaled_predicted_prices = my_outputs_1.cpu().detach().numpy()
predicted_prices_1 = scaler.inverse_transform(scaled_predicted_prices)
print(predicted_prices_1)

#build a second LSTM model for only one ticker at a time
model_1 = LSTM(input_size=1, hidden_size=100, num_layers=1, output_size=1).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model_1.parameters(), lr=0.002)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

#train LSTM model (for only one ticker at a time) on train_1 data(till data_length -20 training days)
num_epochs = 10
for epoch in range(num_epochs):
  for i in range(int(len(tickers))):
        # Slice data for current ticker
        X_train_i = X_train_1[:, :,i:i+1]
        y_train_i = y_train_1[:,i:i+1]
        model_1.train()
        outputs = model_1(X_train_i.float().to(device))
        optimizer.zero_grad()
        loss = criterion(outputs, y_train_i.float().to(device))
        loss.backward()
        optimizer.step()
        scheduler.step()
  if (epoch+1) % 2 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

#Freeze the first three LSTM layers
for param in model_1.lstm1.parameters():
    param.requires_grad = False
for param in model_1.lstm2.parameters():
    param.requires_grad = False
for param in model_1.lstm3.parameters():
    param.requires_grad = False

#train the LSTM model (for only one ticker at a time) with frozen layers on train_2 data(from data_length 20 to data_length 5)and evaluate immediately
my_outputs_1f = torch.zeros(5,10).to(device)
for epoch in range(num_epochs):
  for i in range(int(len(tickers))):
        # Slice data for current ticker
        X_train_i = X_train_2[:, :,i:i+1]
        y_train_i = y_train_2[:,i:i+1]
        model_1.train()
        outputs = model_1(X_train_i.float().to(device))
        optimizer.zero_grad()
        loss = criterion(outputs, y_train_i.float().to(device))
        loss.backward()
        optimizer.step()
        scheduler.step()
        model_1.eval()
        my_local_outputs = model_1(X_test[:, :, i:i+1].float().to(device))
        # Process my_outputs for current ticker
        my_outputs_1f[:, i:i+1] = my_local_outputs
  if (epoch+1) % 2 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
test_loss = criterion(my_outputs_1f, y_test.float().to(device))
print(f'Test Loss: {test_loss.item():.4f}')
print(f'Test RMSE: {math.sqrt(test_loss.item()):.4f}')
print(my_outputs_1f)

#predict future prices
scaled_predicted_prices = my_outputs_1f.cpu().detach().numpy()
predicted_prices_1f = scaler.inverse_transform(scaled_predicted_prices)
print(predicted_prices_1f)

#Plot results (check the similarty first)
plt.figure(figsize=(14,7))
plt.plot(df.index[-5:], df['UPS'][-5:], label='Actual APPL Prices')
plt.plot(df.index[-5:], predicted_prices_0[:, tickers.index('UPS')], label='Predicted ADI Prices')
plt.plot(df.index[-5:], predicted_prices_0f[:, tickers.index('UPS')], label='Predicted ADI Prices 0f')
plt.plot(df.index[-5:], predicted_prices_1[:, tickers.index('UPS')], label='Predicted ADI Prices 1')
plt.plot(df.index[-5:], predicted_prices_1f[:, tickers.index('UPS')], label='Predicted ADI Prices 1f')
plt.legend()
plt.ylim(bottom=100)
plt.show()

plt.figure(figsize=(14,7))
plt.plot(df.index[-5:], df['KMB'][-5:], label='Actual PEP Prices')
plt.plot(df.index[-5:], predicted_prices_0[:, tickers.index('KMB')], label='Predicted PEP Prices')
plt.plot(df.index[-5:], predicted_prices_0f[:, tickers.index('KMB')], label='Predicted PEP Prices 0f')
plt.plot(df.index[-5:], predicted_prices_1[:, tickers.index('KMB')], label='Predicted PEP Prices 1')
plt.plot(df.index[-5:], predicted_prices_1f[:, tickers.index('KMB')], label='Predicted PEP Prices 1f')
plt.legend()
plt.ylim(bottom=100)
plt.show()







