import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# read the data from csv files
train_set = pd.read_csv('train.csv').dropna()
test_set = pd.read_csv('test.csv').dropna()

# Prepare the features (X) and target variables (y)
# of the training set and test set for the linear regression model.
X_train = train_set[[f'newclose_{i}day' for i in range(1, 6)]]
y_train = train_set['close']
X_test = test_set[[f'newclose_{i}day' for i in range(1, 6)]]
y_test = test_set['close']


# create a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# predict the target variable for the test set
y_predict = model.predict(X_test)

# Performance standard
r2 = r2_score(y_test, y_predict)
mse = mean_squared_error(y_test, y_predict)


# Output
print(f"R-square value: {r2:.4f}")
print(f"Mean squared error: {mse:.6f}")

# Regression equation
print("\n Regression equation:")
# y = b0 + b1*x1 + b2*x2 + ... + bn*xn
coefficients = model.coef_
intercept = model.intercept_
equation = f"Closed price of Apple = {intercept : .4f}"
for i, coef in enumerate(coefficients, 1): # start from 1, i.e. newclose_1day
    equation += f" + {coef : .4f}*newclose_{i}day"

print(equation)

# Strict answer to user's question
print("\n The coefficient before the price of the previous day (newclose_1day) is:", f"{coefficients[0] : .4f}")