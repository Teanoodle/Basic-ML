import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression as LR
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
from sklearn.tree import DecisionTreeRegressor as DTR

sp500 = pd.read_csv('all_stocks_5yr.csv')
# Fill in Missing Values, use the forward fill method.
sp500.fillna(method='ffill', inplace=True)

# Check if there are any missing values.
sp500.isnull().sum()
apple_data = sp500[sp500['Name'] == 'AAPL']
apple = sp500[sp500['Name'] == 'AAPL'].copy()
apple = apple.sort_values('date')  # Sort by date
raw_df = apple.copy()

sp500['date'] = pd.to_datetime(sp500['date'])   # Convert the date column to datetime format

start_date = sp500['date'].min()   # Earliest date
end_date = sp500['date'].max()    # Latest date



class Imputer(BaseEstimator, TransformerMixin):
    """An Imputater: 
    It fills in the missing data with the average of 
    the previous day and the next day. 
    """
    def __init__(self, method='linear', limit_direction='both'):
        self.method = method
        self.limit_direction = limit_direction

    def fit(self, X, y=None):
        # No statistics need to be calculated in advance, 
        # since it depends on the adjacent values of X.
        self.is_fitted_ = True
        return self
    def transform(self, X):
        # Use the linear interpolation method to interpolate X
        # 'limit_direction' parameter is set to 'both', 
        # which is the average of both sides of the missing value.
        return X.interpolate(method =self.method, limit_direction =self.limit_direction)
    
# I wrote a file to conduct an unit test on Imputater and named it test1.py

class Attributesadder(BaseEstimator, TransformerMixin):
    """An Attributesadder:
    It adds the prices of previous 5 days as new attributes.
    """
    def __init__(self, previousday=5):
        self.previousday = previousday

    def fit(self, X, y=None):
        # Feature addition does not require pre-calculated parameters
        self.is_fitted_ = True
        return self
    def transform(self, X):
        X = X.copy()
        for i in range(1, 6):
            X[f'newclose_{i}day'] = X['close'].shift(i)
        return X.dropna()
        # Delete the rows containing missing values
        # because the data of the first 5 days is unavailable at the beginning of the dataset.

class Scaler(BaseEstimator, TransformerMixin):
    """An Scaler: 
    It scales everything numerical data into values between 0 and 1.
    """
    def __init__(self):
        self.scaler = MinMaxScaler(feature_range = (0, 1) )
    #The transformation is given by:
    # X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
    # X_scaled = X_std * (max - min) + min
    # where min, max = feature_range.
    def fit(self, X, y=None):
        # Call MinMaxScaler to calculate and store:
        # The minimum and the maximum value of each feature
        self.scaler.fit(X)
        return self
    def transform(self, X):
        return self.scaler.transform(X)




# Create a pipeline to process the data
numeric_features = ['open', 'high', 'low', 'close', 'volume']
pipeline = Pipeline([
    ('Step1', Imputer()),  
    ('Step2', Attributesadder()),
    ('Step3', Scaler())
])

from sklearn.model_selection import TimeSeriesSplit
original_5 = ['open', 'high', 'low', 'close', 'volume']
clean_original_5 = pipeline.fit_transform(raw_df[original_5])

# Convert the values back to a DataFrame
clean_df = pd.DataFrame(clean_original_5, 
                       columns=original_5 + [f'newclose_{i}day' for i in range(1,6)])
clean_df['date'] = pd.to_datetime(raw_df['date'].iloc[5:].reset_index(drop=True))
clean_df['Stock_name'] = 'AAPL'  # add a column for the stock name

# In order to predict the price on the fifth day in the future, 
# we need to move the target variable backward by five days
clean_df['future5_close'] = clean_df['close'].shift(-5)

# Delete the lines containing NaN values.
# Since the shift operation, the future_close in the last 5 lines will be NaN.
clean_df.dropna(inplace=True)

# Split the data into training and testing sets
tscv = TimeSeriesSplit(n_splits=2, test_size=int(len(clean_df)*0.2)) # 80% of the data is used for training and 20% for testing
train_index, test_index = list(tscv.split(clean_df))[-1]

# Here, I save the data of train and test in memory 
# instead of generating new csv files, just for convenience.
train_set = clean_df.iloc[train_index]
test_set = clean_df.iloc[test_index]

# Let's do some basic checks, hopefully they are correct ! ! ! ! ! ! ! ! ! ! ! ! !
print(f"Train date: {clean_df.iloc[train_index].date.min()} to {clean_df.iloc[train_index].date.max()}")
print(f"Test date: {clean_df.iloc[test_index].date.min()} to {clean_df.iloc[test_index].date.max()}")
print("=="*50)



# Prepare the features (X) and target variables (y)
# of the training set and test set for the linear regression model.
X_train = train_set[[f'newclose_{i}day' for i in range(1, 6)]]
y_train = train_set['future5_close']
X_test = test_set[[f'newclose_{i}day' for i in range(1, 6)]]
y_test = test_set['future5_close']

# create a linear regression model
model = LR()
model.fit(X_train, y_train)

# predict the target variable for the test set
y_predict = model.predict(X_test)

# Performance standard
r2 = r2_score(y_test, y_predict)
mse = mean_squared_error(y_test, y_predict)


# Output
print("\n Regression Performance:")
print(f"R-square value: {r2 : .4f}")
print(f"Mean squared error: {mse : .6f}")

# Regression equation
print("\n Regression equation:")
# y = b0 + b1*x1 + b2*x2 + ... + bn*xn
coefficients = model.coef_
intercept = model.intercept_
equation = f"Closed price of Apple at 5th day= {intercept : .4f}"
for i, coef in enumerate(coefficients, 1): # start from 1, i.e. newclose_1day
    equation += f" + {coef : .4f}*newclose_{i}day"

print(equation)

# Strict answer to user's question
print("\n The coefficient before the price of the previous day (newclose_1day) is:", f"{coefficients[0] : .4f}")


print("=="*50)
# Create decision tree regressor
dt_regressor = DTR(random_state=42, max_depth=5)  # Set a random state for reproducibility and limit the depth of the tree to avoid overfitting
dt_regressor.fit(X_train, y_train)

# Prediction
y_predict = dt_regressor.predict(X_test)


r2 = r2_score(y_test, y_predict)
mse = mean_squared_error(y_test, y_predict)
features = [f'newclose_{i}day' for i in range(1, 6)]
feature_importance = dt_regressor.feature_importances_


# Output
print("\n Decision Tree Performance:")
print(f"R-square value: {r2 : .4f}")
print(f"Mean squared error: {mse : .6f}")
print("\n Features importance:")
for i, importance in enumerate(feature_importance):
    print(f"{features[i]}: {importance : .4f}")