import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler

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
X = data.drop(['Date', 'Henry Hub Natural Gas Spot Price Dollars per Million Btu', 'PriceDiff'], axis=1)
y = data['Henry Hub Natural Gas Spot Price Dollars per Million Btu']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the parameter grid for GridSearchCV
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Initialize Random Forest model
rf_model = RandomForestRegressor(random_state=42)

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, scoring='neg_root_mean_squared_error')

# Fit the grid search to the data
grid_search.fit(X_train_scaled, y_train)

# Get the best parameters
best_params = grid_search.best_params_
print("Best Parameters:", best_params)

# Reinitialize Random Forest model with best parameters
best_rf_model = RandomForestRegressor(**best_params, random_state=42)

# Fit the model with best parameters to the training data
best_rf_model.fit(X_train_scaled, y_train)

# Make predictions
train_predictions = best_rf_model.predict(X_train_scaled)
test_predictions = best_rf_model.predict(X_test_scaled)

# Calculate RMSE
train_rmse = np.sqrt(mean_squared_error(y_train, train_predictions))
test_rmse = np.sqrt(mean_squared_error(y_test, test_predictions))

print('Train RMSE:', train_rmse)
print('Test RMSE:', test_rmse)
