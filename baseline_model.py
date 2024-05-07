import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

# Load the dataset
data = pd.read_csv('/Users/danielbrown/Desktop/Henry_Hub_Natural_Gas_Spot_Price_1.csv')

# Feature Engineering (if not already done)
data['Price_Lag_1'] = data['Henry Hub Natural Gas Spot Price Dollars per Million Btu'].shift(1)
data['Price_Lag_2'] = data['Henry Hub Natural Gas Spot Price Dollars per Million Btu'].shift(2)
data['Price_Lag_3'] = data['Henry Hub Natural Gas Spot Price Dollars per Million Btu'].shift(3)
data['Price_Lag_4'] = data['Henry Hub Natural Gas Spot Price Dollars per Million Btu'].shift(4)
data['Price_Lag_5'] = data['Henry Hub Natural Gas Spot Price Dollars per Million Btu'].shift(5)
data['Price_MA'] = data['Henry Hub Natural Gas Spot Price Dollars per Million Btu'].rolling(window=5).mean()
data['Day_of_Week'] = pd.to_datetime(data['Date']).dt.dayofweek
data['Month'] = pd.to_datetime(data['Date']).dt.month
data['Year'] = pd.to_datetime(data['Date']).dt.year

# Drop rows with missing values created by lag features and rolling mean
data.dropna(inplace=True)

# Prepare data
prices = data['Henry Hub Natural Gas Spot Price Dollars per Million Btu'].values.astype(float)
prices_diff = data['PriceDiff'].values.astype(float)

# Split the dataset into training and testing sets
train_size = int(len(data) * 0.80)
train, test = data[:train_size], data[train_size:]

# Compute mean of the target variable in the training set
mean_price = np.mean(train['Henry Hub Natural Gas Spot Price Dollars per Million Btu'])

# Predict using mean for all instances in the test set
baseline_predictions = np.full_like(test['Henry Hub Natural Gas Spot Price Dollars per Million Btu'], fill_value=mean_price)

# Calculate RMSE for the baseline model
baseline_rmse = np.sqrt(mean_squared_error(test['Henry Hub Natural Gas Spot Price Dollars per Million Btu'], baseline_predictions))
print('Baseline RMSE:', baseline_rmse)
