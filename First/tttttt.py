import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
sp500 = pd.read_csv('all_stocks_5yr.csv')

# imputater did in question 8
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler
def impute_with_avg(df, cols):
    for col in cols:
        vals = df[col]
        mask = vals.isnull()
        pre = vals.shift(1)
        nxt = vals.shift(-1)
        mean = (pre + nxt) / 2
        df.loc[mask, col] = mean[mask]
    return df

class AttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self,n_days=5,base_col='close'):
        self.n_days=n_days
        self.base_col=base_col
    def fit(self,X,y=None):
        return self
    def transform(self, X):
        X_new = X.copy()
        for i in range(1, self.n_days+1):
            X_new[f'{self.base_col}_lag_{i}'] = X_new[self.base_col].shift(i)
        X_new = X_new.dropna()
        return X_new
scaler = MinMaxScaler()
cols_to_impute = ['open', 'high', 'low']
apple_data = sp500[sp500['Name'] == 'AAPL']
(apple_data, cols_to_impute)

adder = AttributesAdder(n_days=5, base_col='close')
apple_data_lag = adder.transform(apple_data)

features = [f'close_lag_{i}' for i in range(1, 6)] + ['open', 'high', 'low', 'close']
X = apple_data_lag[features]
X_scaled = scaler.fit_transform(X)

y = apple_data_lag['close']

print(pd.DataFrame(X_scaled, columns=features).head())

from sklearn.model_selection import TimeSeriesSplit
t=TimeSeriesSplit(n_splits=10)
for train_index, test_index in t.split(X):
    print("Train:",train_index,"Test:",test_index)
    X_train,X_test=X.iloc[train_index],X.iloc[test_index]
    y_train,y_test=y.iloc[train_index],y.iloc[test_index]

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
apple_data = sp500[sp500['Name'] == 'AAPL'].sort_values('date').reset_index(drop=True)
apple_data = impute_with_avg(apple_data, ['open','high','low'])
lr=LinearRegression()
lr.fit(X_train,y_train)
y_pre=lr.predict(X_test)
MSE=mean_squared_error(y_test,y_pre)
r2=r2_score(y_test,y_pre)
print(f"R-square:{r2:.4f}")
print(f"MSE:{MSE:.4f}")
print("Intercept:",lr.intercept_)
for name, coef in zip(features, lr.coef_):
    print(f"Feature: {name}, Coefficient: {coef:.4f}")
print("the last close price:", lr.coef_[features.index('close_lag_1')])


from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

apple_data = sp500[sp500['Name'] == 'AAPL'].sort_values('date').reset_index(drop=True)
apple_data = impute_with_avg(apple_data, ['open','high','low'])
tree = DecisionTreeRegressor(random_state=42)
tree.fit(X_train, y_train)
y_pred = tree.predict(X_test)
print(f"R-square: {r2_score(y_test, y_pred):.4f}")
print(f"MSE: {mean_squared_error(y_test, y_pred):.6f}")
importances = tree.feature_importances_
for name, score in zip(features, importances):
    print(f"Feature: {name}, Importance: {score:.4f}")