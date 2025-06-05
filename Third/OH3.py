import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Load S&P500 stock data
sp500 = pd.read_csv('all_stocks_5yr.csv')


# Convert close price to numeric
sp500['close'] = pd.to_numeric(sp500['close'])

# Calculate log return with transform to maintain index alignment
sp500['log_return'] = np.log(sp500['close']) - np.log(sp500.groupby('Name')['close'].transform('shift'))
log_return_series = sp500.dropna(subset=['log_return'])
print("Log Return Series:")
print(log_return_series)

#--------------------------------------------------------------------------------------------
# ADF test on raw close prices
result = adfuller(sp500['close'].dropna(), maxlag=1)
adf_statistic = result[0]
p_value = result[1]
critical_values = result[4]

print('\nADF Test Results (Raw Close Prices):')
print(f'ADF Statistic: {adf_statistic}')
print(f'p-value: {p_value}')
print('Critical Values:')
for key, value in critical_values.items():
    print(f'   {key}: {value}')

#--------------------------------------------------------------------------------------------
# ADF test on log returns
result = adfuller(log_return_series['log_return'], maxlag=1) 
adf_statistic = result[0]
p_value = result[1]
critical_values = result[4]

print('\nADF Test Results (Log Returns):')
print(f'ADF Statistic: {adf_statistic}')
print(f'p-value: {p_value}')
print('Critical Values:')
for key, value in critical_values.items():
    print(f'   {key}: {value}')
#--------------------------------------------------------------------------------------------



# Perform EDA
# display the basic information of the dataset
print(sp500.info())

# Display summary statistics
print(sp500.describe())

# Sample a few stocks for visualization
stocks_to_plot = sp500['Name'].sample(5, random_state=1).tolist()

# Filter data for the sampled stocks - only contain "name" and the ralated data
sampled_data = sp500[sp500['Name'].isin(stocks_to_plot)]

# Plotting stock prices over time for the sampled stocks
plt.figure(figsize=(14, 7))
for stock in stocks_to_plot:
    stock_data = sampled_data[sampled_data['Name'] == stock]
    plt.plot(stock_data['date'], stock_data['close'], label=stock)
plt.title('Stock Prices Over Time')
plt.xlabel('Date')
plt.ylabel('Closing Price')


# Formatting the x-axis for better readability
# Setting the major ticks on the x-axis to be at the start of each year
plt.gca().xaxis.set_major_locator(mdates.YearLocator())
# display the year and month
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
# Rotates the x-axis by 45 degrees for better readability
plt.xticks(rotation=45)

plt.legend()
plt.tight_layout()
plt.show()

# Function to calculate Relative Strength Index (RSI) -Self defination
def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# load the data again
sp500 = pd.read_csv('all_stocks_5yr.csv')
# Ensure 'date' is in datetime format
sp500['date'] = pd.to_datetime(sp500['date'])
# Sort the data by 'Name' and 'date'
sp500.sort_values(by=['Name', 'date'])
# Calculate moving averages
sp500['MA_5'] = sp500.groupby('Name')['close'].transform(lambda x: x.rolling(window=5).mean())
sp500['MA_10'] = sp500.groupby('Name')['close'].transform(lambda x: x.rolling(window=10).mean())
sp500['MA_30'] = sp500.groupby('Name')['close'].transform(lambda x: x.rolling(window=30).mean())
sp500['MA_60'] = sp500.groupby('Name')['close'].transform(lambda x: x.rolling(window=60).mean())

# Calculate RSI
sp500['RSI'] = sp500.groupby('Name')['close'].transform(lambda x: calculate_rsi(x))

# Calculate On-Balance Volume (OBV)
sp500['OBV'] = sp500.groupby('Name').apply(lambda x: (np.sign(x['close'].diff()) * x['volumn']).cumsum())

# Display the first few rows of the DataFrame
print("\nTechnical Indicators Sample:")
print(sp500[['Name', 'date', 'close', 'MA_5', 'MA_10', 'RSI', 'OBV']].head())
#--------------------------------------------------------------------------------------------



