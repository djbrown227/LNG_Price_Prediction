import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error
from sklearn.base import BaseEstimator
from keras.wrappers.scikit_learn import KerasRegressor

# Load the dataset
data = pd.read_csv('/Users/danielbrown/Desktop/Henry_Hub_Natural_Gas_Spot_Price_1.csv')

# Feature Engineering
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

# Normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
prices_scaled = scaler.fit_transform(prices.reshape(-1, 1))

# Combine normalized prices with price differences
data_scaled = np.column_stack((prices_scaled, prices_diff))

# Split the dataset into training and testing sets
train_size = int(len(data_scaled) * 0.80)
train, test = data_scaled[:train_size], data_scaled[train_size:]

# Convert an array of values into a dataset matrix
def create_dataset(dataset, time_step=1):
    X, Y = [], []
    for i in range(len(dataset) - time_step - 1):
        a = dataset[i:(i + time_step), :]
        X.append(a)
        Y.append(dataset[i + time_step, 0])
    return np.array(X), np.array(Y)

# Reshape into X=t and Y=t+1
time_step = 10
X_train, Y_train = create_dataset(train, time_step)
X_test, Y_test = create_dataset(test, time_step)

# Reshape input to be [samples, time steps, features]
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 2)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 2)

# Define the LSTM model
def create_model(units=50):
    model = Sequential([
        LSTM(units=units, return_sequences=True, input_shape=(time_step, 2)),
        LSTM(units=units),
        Dense(units=1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Custom class to wrap Keras model for scikit-learn compatibility
class KerasRegressorWrapper(BaseEstimator):
    def __init__(self, build_fn=create_model, **kwargs):
        self.build_fn = build_fn
        self.kwargs = kwargs
        self.model = None

    def fit(self, X, y):
        self.model = self.build_fn(**self.kwargs)
        self.model.fit(X, y, verbose=0)
        return self

    def predict(self, X):
        return self.model.predict(X)

# Define the grid search parameters
param_grid = {
    'build_fn': [create_model],
    'units': [50, 100],
    'epochs': [50, 100],
    'batch_size': [64, 128]
}

# Perform GridSearchCV
grid_search = GridSearchCV(estimator=KerasRegressorWrapper(), param_grid=param_grid, cv=3, scoring='neg_mean_squared_error', verbose=2)
grid_search.fit(X_train, Y_train)

# Best parameters
best_params = grid_search.best_params_
print("Best Parameters:", best_params)

# Re-train the model with best parameters
best_model = create_model(units=best_params['units'])
best_model.fit(X_train, Y_train, epochs=best_params['epochs'], batch_size=best_params['batch_size'], verbose=0)

# Make predictions
train_predict = best_model.predict(X_train)
test_predict = best_model.predict(X_test)

# Inverse transform the predictions
train_predict = scaler.inverse_transform(np.concatenate((train_predict, X_train[:, -1, 1].reshape(-1, 1)), axis=1))[:, 0]
test_predict = scaler.inverse_transform(np.concatenate((test_predict, X_test[:, -1, 1].reshape(-1, 1)), axis=1))[:, 0]

# Convert Y_train and Y_test back to original scale for RMSE calculation
Y_train_orig = scaler.inverse_transform(Y_train.reshape(-1, 1))[:, 0]
Y_test_orig = scaler.inverse_transform(Y_test.reshape(-1, 1))[:, 0]

# Calculate RMSE
train_rmse = np.sqrt(mean_squared_error(Y_train_orig, train_predict))
test_rmse = np.sqrt(mean_squared_error(Y_test_orig, test_predict))
print('Train RMSE:', train_rmse)
print('Test RMSE:', test_rmse)
