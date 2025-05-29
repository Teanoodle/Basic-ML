import pandas as pd
from sklearn.tree import DecisionTreeRegressor as DTR
from sklearn.metrics import r2_score, mean_squared_error

# read the data from csv files
train_set = pd.read_csv('train.csv').dropna()
test_set = pd.read_csv('test.csv').dropna()

X_train = train_set[[f'newclose_{i}day' for i in range(1, 6)]]
y_train = train_set['close']
X_test = test_set[[f'newclose_{i}day' for i in range(1, 6)]]
y_test = test_set['close']

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
print(f"R-square value: {r2 : .4f}")
print(f"Mean squared error: {mse : .6f}")
print("\n Features importance:")
for i, importance in enumerate(feature_importance):
    print(f"{features[i]}: {importance : .4f}")