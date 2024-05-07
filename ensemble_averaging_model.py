import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.svm import SVR
from keras.models import Sequential
from keras.layers import LSTM, Dense
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.python.keras.layers import Dense
#import tensorflow as tf

# Use tf.keras to reference Keras modules
#models = tf.keras.models
#layers = tf.keras.layers
#optimizers = tf.keras.optimizers
# Import TimeseriesGenerator from tf.keras
#TimeseriesGenerator = tf.keras.preprocessing.sequence.TimeseriesGenerator



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

# Prepare data for LSTM
prices = data['Henry Hub Natural Gas Spot Price Dollars per Million Btu'].values.astype(float)
prices_diff = data['PriceDiff'].values.astype(float)

scaler_lstm = StandardScaler()
prices_scaled = scaler_lstm.fit_transform(prices.reshape(-1, 1))

# Combine normalized prices with price differences
data_scaled = np.column_stack((prices_scaled, prices_diff))

# Split the dataset into training and testing sets for LSTM
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

# Reshape into X=t and Y=t+1 for LSTM
time_step = 10
X_train_lstm, Y_train_lstm = create_dataset(train, time_step)
X_test_lstm, Y_test_lstm = create_dataset(test, time_step)

# Reshape input to be [samples, time steps, features]
X_train_lstm = X_train_lstm.reshape(X_train_lstm.shape[0], X_train_lstm.shape[1], 2)
X_test_lstm = X_test_lstm.reshape(X_test_lstm.shape[0], X_test_lstm.shape[1], 2)

# Build the LSTM model
model_lstm = Sequential([
    LSTM(units=50, return_sequences=True, input_shape=(time_step, 2)),
    LSTM(units=50),
    Dense(units=1)
])
model_lstm.compile(optimizer='adam', loss='mean_squared_error')

# Train the LSTM model
model_lstm.fit(X_train_lstm, Y_train_lstm, epochs=100, batch_size=64)

# Make predictions with LSTM model
train_predict_lstm = model_lstm.predict(X_train_lstm)
test_predict_lstm = model_lstm.predict(X_test_lstm)

# Inverse transform the predictions
train_predict_lstm = scaler_lstm.inverse_transform(np.concatenate((train_predict_lstm, X_train_lstm[:, -1, 1].reshape(-1, 1)), axis=1))[:, 0]
test_predict_lstm = scaler_lstm.inverse_transform(np.concatenate((test_predict_lstm, X_test_lstm[:, -1, 1].reshape(-1, 1)), axis=1))[:, 0]

# Load the dataset again for other models
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

# Prepare data for other models
X_other = data.drop(['Date', 'Henry Hub Natural Gas Spot Price Dollars per Million Btu', 'PriceDiff'], axis=1)
y_other = data['Henry Hub Natural Gas Spot Price Dollars per Million Btu']

# Split the dataset into training and testing sets for other models
X_train_other, X_test_other, y_train_other, y_test_other = train_test_split(X_other, y_other, test_size=0.2, random_state=42)

# Standardize the features for other models
scaler_other = StandardScaler()
X_train_scaled_other = scaler_other.fit_transform(X_train_other)
X_test_scaled_other = scaler_other.transform(X_test_other)

# Initialize and train other models
gbm_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
gbm_model.fit(X_train_scaled_other, y_train_other)

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled_other, y_train_other)

svr_model = SVR(kernel='rbf')
svr_model.fit(X_train_scaled_other, y_train_other)

# Make predictions with other models
train_predictions_gbm = gbm_model.predict(X_train_scaled_other)
test_predictions_gbm = gbm_model.predict(X_test_scaled_other)

train_predictions_rf = rf_model.predict(X_train_scaled_other)
test_predictions_rf = rf_model.predict(X_test_scaled_other)

train_predictions_svr = svr_model.predict(X_train_scaled_other)
test_predictions_svr = svr_model.predict(X_test_scaled_other)

# Trim longer predictions arrays to match the shortest length
min_length = min(len(test_predict_lstm), len(test_predictions_gbm), len(test_predictions_rf), len(test_predictions_svr))
test_predict_lstm = test_predict_lstm[:min_length]
test_predictions_gbm = test_predictions_gbm[:min_length]
test_predictions_rf = test_predictions_rf[:min_length]
test_predictions_svr = test_predictions_svr[:min_length]
y_test_other = y_test_other[:min_length]

# Calculate RMSE for individual models
gbm_rmse = np.sqrt(mean_squared_error(y_test_other, test_predictions_gbm))
rf_rmse = np.sqrt(mean_squared_error(y_test_other, test_predictions_rf))
svr_rmse = np.sqrt(mean_squared_error(y_test_other, test_predictions_svr))
lstm_rmse = np.sqrt(mean_squared_error(y_test_other, test_predict_lstm))

# Calculate MAE for individual models
gbm_mae = mean_absolute_error(y_test_other, test_predictions_gbm)
rf_mae = mean_absolute_error(y_test_other, test_predictions_rf)
svr_mae = mean_absolute_error(y_test_other, test_predictions_svr)
lstm_mae = mean_absolute_error(y_test_other, test_predict_lstm)

# Calculate MAPE for individual models
gbm_mape = mean_absolute_percentage_error(y_test_other, test_predictions_gbm)
rf_mape = mean_absolute_percentage_error(y_test_other, test_predictions_rf)
svr_mape = mean_absolute_percentage_error(y_test_other, test_predictions_svr)
lstm_mape = mean_absolute_percentage_error(y_test_other, test_predict_lstm)

# Calculate R2 score for individual models
gbm_r2 = r2_score(y_test_other, test_predictions_gbm)
rf_r2 = r2_score(y_test_other, test_predictions_rf)
svr_r2 = r2_score(y_test_other, test_predictions_svr)
lstm_r2 = r2_score(y_test_other, test_predict_lstm)

# Ensemble predictions
ensemble_test_predictions = (test_predictions_gbm + test_predictions_rf + test_predictions_svr + test_predict_lstm) / 4

# Calculate RMSE for ensemble model
ensemble_test_rmse = np.sqrt(mean_squared_error(y_test_other, ensemble_test_predictions))
# Calculate MAE for ensemble model
ensemble_mae = mean_absolute_error(y_test_other, ensemble_test_predictions)
# Calculate MAPE for ensemble model
ensemble_mape = mean_absolute_percentage_error(y_test_other, ensemble_test_predictions)
# Calculate R2 score for ensemble model
ensemble_r2 = r2_score(y_test_other, ensemble_test_predictions)

# Print the comparison results
print("Model Comparison Results:")
print("Individual Model RMSE:")
print("GBM:", gbm_rmse)
print("Random Forest:", rf_rmse)
print("SVR:", svr_rmse)
print("LSTM:", lstm_rmse)
print("\nIndividual Model MAE:")
print("GBM:", gbm_mae)
print("Random Forest:", rf_mae)
print("SVR:", svr_mae)
print("LSTM:", lstm_mae)
print("\nIndividual Model MAPE:")
print("GBM:", gbm_mape)
print("Random Forest:", rf_mape)
print("SVR:", svr_mape)
print("LSTM:", lstm_mape)
print("\nIndividual Model R2 Score:")
print("GBM:", gbm_r2)
print("Random Forest:", rf_r2)
print("SVR:", svr_r2)
print("LSTM:", lstm_r2)
print("\nEnsemble Model Performance:")
print("Ensemble RMSE:", ensemble_test_rmse)
print("Ensemble MAE:", ensemble_mae)
print("Ensemble MAPE:", ensemble_mape)
print("Ensemble R2 Score:", ensemble_r2)
