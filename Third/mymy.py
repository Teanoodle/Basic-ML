import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Input, LSTM, Dropout, Dense
from sklearn.preprocessing import MinMaxScaler

# Load S&P500 stock data
sp500 = pd.read_csv('all_stocks_5yr.csv')
print(sp500['close'].dtype)
# Determine how many stock companies there are
num_companies = sp500['Name'].nunique()
print(f"Number of unique stock companies: {num_companies}")


# Convert close price to numeric
sp500['close'] = pd.to_numeric(sp500['close'])

# # ADF test on raw close prices
# print("\nADF Test on Raw Close Prices:")
# for name, group in sp500.groupby('Name'):
#     result = adfuller(group['close'].dropna(), maxlag=1)
#     print(f"Stock: {name}")
#     print(f"ADF Statistic: {result[0]:.4f}")
#     print(f"p-value: {result[1]:.4f}")
#     print(f"Critical Values: {result[4]}\n")

# Calculate log return with transform to maintain index alignment
sp500['log_return'] = np.log(sp500['close']) - np.log(sp500.groupby('Name')['close'].transform('shift'))
log_return_series = sp500.dropna(subset=['log_return'])
print("Log Return Series:")
print(log_return_series)

# Group ADF test by stock name with optimized output
# def adf_test(x):
#     result = adfuller(x.dropna(), maxlag=1)
#     return pd.Series({
#         'ADF Statistic': result[0],
#         'p-value': result[1],
#         '1% Critical': result[4]['1%'],
#         '5% Critical': result[4]['5%'],
#         '10% Critical': result[4]['10%']
#     })

# print("\nADF Test Results by Stock:")
# adf_results = log_return_series.groupby('Name')['log_return'].apply(adf_test)
# print(adf_results.reset_index())



# ----------------------------------------------------------------------------------------------------------------------------
# Calculate technical indicators
def calculate_technical_indicators(df):
    # Moving Averages
    df['MA_5'] = df.groupby('Name')['close'].transform(lambda x: x.rolling(5).mean())
    df['MA_10'] = df.groupby('Name')['close'].transform(lambda x: x.rolling(10).mean())
    df['MA_30'] = df.groupby('Name')['close'].transform(lambda x: x.rolling(30).mean())
    df['MA_60'] = df.groupby('Name')['close'].transform(lambda x: x.rolling(60).mean())
    
    # Relative Strength Index (RSI)
    delta = df.groupby('Name')['close'].transform(lambda x: x.diff())
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.groupby(df['Name']).transform(lambda x: x.rolling(14).mean())
    avg_loss = loss.groupby(df['Name']).transform(lambda x: x.rolling(14).mean())
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # On-Balance Volume (OBV)
    df['OBV'] = df.groupby('Name', group_keys=False).apply(
        lambda x: (np.sign(x['close'].diff()) * x['volume']).fillna(0).cumsum()
    )
    
    return df

# Calculate and show technical indicators
sp500_with_indicators = calculate_technical_indicators(sp500)
print("\nTechnical Indicators Sample:")
print(sp500_with_indicators[['Name', 'date', 'close', 'MA_5', 'MA_10', 'RSI', 'OBV']].head())

# Split data into training (80%) and testing (20%) sets while preserving time order
def time_series_split(df, test_size=0.2):
    # Sort by stock name and date
    df = df.sort_values(['Name', 'date'])
    
    # Group by stock name
    grouped = df.groupby('Name')
    
    # Split each stock's data
    train_dfs = []
    test_dfs = []
    
    for name, group in grouped:
        # Calculate split index
        split_idx = int(len(group) * (1 - test_size))
        
        # Split while preserving time order
        train = group.iloc[:split_idx]
        test = group.iloc[split_idx:]
        
        train_dfs.append(train)
        test_dfs.append(test)
    
    return pd.concat(train_dfs), pd.concat(test_dfs)

# Split the data
train_data, test_data = time_series_split(sp500_with_indicators, test_size=0.2)

# Verify the split
print(f"\nTraining set size: {len(train_data)} ({len(train_data)/len(sp500_with_indicators):.1%})")
print(f"Testing set size: {len(test_data)} ({len(test_data)/len(sp500_with_indicators):.1%})")
print("\nSample training data:")
print(train_data[['Name', 'date']].head())
print("\nSample testing data:")
print(test_data[['Name', 'date']].head())

# Data preprocessing
def clean_data(df):
    # Fill missing values by forward and backward filling within each stock group
    df = df.groupby('Name', group_keys=False).apply(lambda x: x.ffill().bfill(), include_groups=False)
    
    # Drop rows where technical indicators couldn't be calculated
    df = df.dropna(subset=['MA_5', 'MA_10', 'MA_30', 'MA_60', 'RSI', 'OBV'])
    
    # Ensure no infinite values
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    
    return df

train_data = clean_data(train_data)
test_data = clean_data(test_data)
# print(f"\n Training set after cleaning: {len(train_data)}")
# print(f"Testing set after cleaning: {len(test_data)}")

# Feature selection for models
features = ['MA_5', 'MA_10', 'MA_30', 'MA_60', 'RSI', 'OBV']
target = 'close'

X_train = train_data[features]
y_train = train_data[target]
X_test = test_data[features]
y_test = test_data[target]

# XGBoost Model

xgb_model = xgb.XGBRegressor(
    objective='reg:squarederror', 
    n_estimators=100,
    random_state=42)
xgb_model.fit(X_train, y_train)

# XGBoost predictions and evaluation
xgb_pred = xgb_model.predict(X_test)
xgb_mse = mean_squared_error(y_test, xgb_pred)
xgb_mae = mean_absolute_error(y_test, xgb_pred)
xgb_rmse = np.sqrt(xgb_mse)
xgb_r2 = r2_score(y_test, xgb_pred)

print("\nXGBoost Model Performance:")
print(f"MSE: {xgb_mse:.4f}")
print(f"MAE: {xgb_mae:.4f}")
print(f"RMSE: {xgb_rmse:.4f}")
print(f"R^2: {xgb_r2:.4f}")

# ——————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Reshape for LSTM [samples, timesteps, features]
X_train_lstm = X_train_scaled.reshape(X_train_scaled.shape[0], 1, X_train_scaled.shape[1])
X_test_lstm = X_test_scaled.reshape(X_test_scaled.shape[0], 1, X_test_scaled.shape[1])

# Build LSTM model
lstm_model = Sequential([
    Input(shape=(1, X_train_scaled.shape[1])),  # Input layer
    LSTM(128, return_sequences=True),  # First layer LSTM
    Dropout(0.2),  # Prevent overfitting
    LSTM(64),  # Second layer LSTM
    Dropout(0.2),  # Dropout second layer
    Dense(32, activation='relu'),  # Fully connected layer
    Dense(1)  # Output layer
])
lstm_model.compile(optimizer='adam', loss='mse')

# Train model
training_log = lstm_model.fit(
                         X_train_lstm, 
                         y_train, 
                         epochs=10, 
                         batch_size=32, 
                         verbose=1)

# Make predictions
lstm_pred = lstm_model.predict(X_test_lstm)

# Evaluate
lstm_mse = mean_squared_error(y_test, lstm_pred)
lstm_mae = mean_absolute_error(y_test, lstm_pred)
lstm_rmse = np.sqrt(lstm_mse)
lstm_r2 = r2_score(y_test, lstm_pred)

# Model evaluation
print("\n LSTM Model Performance:")
print(f"{'MSE:  '}{lstm_mse:.4f}")
print(f"{'MAE:  '}{lstm_mae:.4f}")
print(f"{'RMSE:  '}{lstm_rmse:.4f}")
print(f"{'R^2:  '}{lstm_r2:.4f}")

# ——————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————


# XGBoost Classifier for Market Direction Prediction
# Create binary label for daily return direction (1 for positive, 0 for negative)
train_data['direction'] = (train_data['log_return'] > 0).astype(int)
test_data['direction'] = (test_data['log_return'] > 0).astype(int)

# Feature selection for classification
X_train_clf = train_data[features]
y_train_clf = train_data['direction']
X_test_clf = test_data[features]
y_test_clf = test_data['direction']

# XGBoost Classifier
xgb_clf = xgb.XGBClassifier(
    objective='binary:logistic',
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
    eval_metric='logloss'
)
xgb_clf.fit(X_train_clf, y_train_clf)

# Classification predictions and evaluation
xgb_clf_pred = xgb_clf.predict(X_test_clf)
xgb_clf_accuracy = (xgb_clf_pred == y_test_clf).mean()

print("\nXGBoost Classifier Performance:")
print(f"Accuracy: {xgb_clf_accuracy:.4f}")

# LSTM Classifier for Market Direction Prediction
# Prepare data for LSTM
scaler = MinMaxScaler()
X_train_clf_scaled = scaler.fit_transform(X_train_clf)
X_test_clf_scaled = scaler.transform(X_test_clf)

# Reshape for LSTM [samples, timesteps, features]
X_train_lstm_clf = X_train_clf_scaled.reshape(X_train_clf_scaled.shape[0], 1, X_train_clf_scaled.shape[1])
X_test_lstm_clf = X_test_clf_scaled.reshape(X_test_clf_scaled.shape[0], 1, X_test_clf_scaled.shape[1])

# Build LSTM classification model
lstm_clf_model = Sequential([
    Input(shape=(1, X_train_clf_scaled.shape[1])),
    LSTM(128, return_sequences=True),
    Dropout(0.2),
    LSTM(64),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')  # Binary classification output
])

lstm_clf_model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train model
training_log2 = lstm_clf_model.fit(
    X_train_lstm_clf,
    y_train_clf,
    epochs=15,
    batch_size=32,
    verbose=1
)

# Evaluate model
lstm_clf_pred_proba = lstm_clf_model.predict(X_test_lstm_clf)
lstm_clf_pred = (lstm_clf_pred_proba > 0.5).astype(int)
lstm_clf_accuracy = (lstm_clf_pred.flatten() == y_test_clf).mean()

print("\nLSTM Classifier Performance:")
print(f"Accuracy: {lstm_clf_accuracy:.4f}")
