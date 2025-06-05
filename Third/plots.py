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
print(sp500['close'].dtype)
# Determine how many stock companies there are
num_companies = sp500['Name'].nunique()
print(f"Number of unique stock companies: {num_companies}")


# Convert close price to numeric
sp500['close'] = pd.to_numeric(sp500['close'])

# Calculate log return with transform to maintain index alignment
sp500['log_return'] = np.log(sp500['close']) - np.log(sp500.groupby('Name')['close'].transform('shift'))
log_return_series = sp500.dropna(subset=['log_return'])
print("Log Return Series:")
print(log_return_series)


# display the basic information of the dataset
print(sp500.info())

# Display summary statistics
print(sp500.describe())


# Enhanced EDA Analysis for AAPL
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from scipy import stats

# Prepare AAPL data
aapl = sp500[sp500['Name'] == 'AAPL'].copy()
aapl['date'] = pd.to_datetime(aapl['date'])
aapl.set_index('date', inplace=True)
aapl_log = aapl[['log_return']].dropna()


plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plt.plot(aapl['close'])
plt.title('AAPL Price Trend')
plt.xlabel('Date')
plt.ylabel('Price')
plt.grid()

plt.subplot(1, 2, 2)
plt.plot(aapl_log['log_return'])
plt.title('AAPL Log Return Trend')
plt.xlabel('Date')
plt.ylabel('Log Return')
plt.grid()
plt.tight_layout()
plt.show()

plt.figure(figsize=(14, 6))
plt.subplot(2, 2, 1)
plt.hist(aapl['close'], bins=30, edgecolor='black')
plt.title('Price Distribution')
plt.xlabel('Price')

plt.subplot(2, 2, 2)
plt.hist(aapl_log['log_return'], bins=30, edgecolor='black')
plt.title('Log Return Distribution')
plt.xlabel('Log Return')

plt.subplot(2, 2, 3)
stats.probplot(aapl['close'], plot=plt)
plt.title('Price Q-Q Plot')

plt.subplot(2, 2, 4)
stats.probplot(aapl_log['log_return'], plot=plt)
plt.title('Log Return Q-Q Plot')
plt.tight_layout()
plt.show()


plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plot_acf(aapl['close'].dropna(), lags=30, title='Price ACF')

plt.subplot(1, 2, 2)
plot_acf(aapl_log['log_return'], lags=30, title='Log Return ACF')
plt.tight_layout()
plt.show()

plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plot_pacf(aapl['close'].dropna(), lags=30, title='Price PACF')

plt.subplot(1, 2, 2)
plot_pacf(aapl_log['log_return'], lags=30, title='Log Return PACF')
plt.tight_layout()
plt.show()


plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
sns.boxplot(data=aapl[['close']])
plt.title('Price Distribution')
plt.xticks(rotation=45)

plt.subplot(1, 2, 2)
sns.boxplot(data=aapl_log[['log_return']])
plt.title('Log Return Distribution')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


z_scores = np.abs(stats.zscore(aapl[['log_return']]))
outliers = aapl[(z_scores > 3).any(axis=1)]
print("\nDetected Outliers:")
print(outliers['log_return'].head())



plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plot_acf(aapl['log_return'].dropna(), lags=30, title='Log Return ACF')
plt.subplot(1, 2, 2)
plot_pacf(aapl['log_return'].dropna(), lags=30, title='Log Return PACF')
plt.tight_layout()
plt.show()

plt.figure(figsize=(18, 6))
plt.subplot(1, 2, 1)
corr_matrix_raw = aapl[['close', 'volume', 'open', 'high', 'low']].corr()
sns.heatmap(corr_matrix_raw, annot=True, cmap='coolwarm', center=0, vmin=-1, vmax=1)
plt.title('Raw Price Feature Correlation')

plt.subplot(1, 2, 2)
corr_matrix_log = aapl[['log_return', 'volume', 'open', 'high', 'low']].corr()
sns.heatmap(corr_matrix_log, annot=True, cmap='coolwarm', center=0, vmin=-1, vmax=1)
plt.title('Log Return Feature Correlation')
plt.tight_layout()
plt.show()

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