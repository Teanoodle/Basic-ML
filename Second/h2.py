import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Question 1: Import data and check
# import datasets
loan_data = pd.read_csv('loan_approval_dataset.csv')

# Missing values check:
print(loan_data.isnull().sum())

# Numerical and categorical columns
nume = loan_data.select_dtypes(include=['int64','float64'])
cate = loan_data.select_dtypes(include=['object'])
print("\n Numerical features:", nume.columns.to_numpy())
print("\n Categorical features:", cate.columns.to_numpy())

# Question 2: Perform EDA
import matplotlib.pyplot as plt
import seaborn as sns

# # Basic statistics
# print(nume.describe())
# print("\n")
# print(cate.describe())


# Categorical features counts
# print("\nPlotting categorical features counts...")
# plt.figure(figsize=(12, 6))
# for i, col in enumerate(cate.columns):
#     plt.subplot(2, 2, i+1)
#     sns.countplot(x=col, data=loan_data)
#     plt.title(f'{col} Distribution')
#     plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()

# # Numerical features correlation
# print("\nPlotting numerical features correlation...")
# plt.figure(figsize=(10, 8))
# sns.heatmap(nume.corr(), annot=True, cmap='coolwarm', center=0)
# plt.title('Numerical Features Correlation')
# plt.show()

# Question 3: Preprocessing pipeline
# Separate features and target
new_set = loan_data.drop(['loan_id'], axis=1)
X = new_set.drop(['loan_status'], axis=1)
y = new_set['loan_status']

# Identify numerical and categorical columns
num_v = X.select_dtypes(include=['int64','float64']).columns
cat_v = X.select_dtypes(include=['object']).columns

# Create separate pipelines
numeric_pipeline = Pipeline([
    ('scaler', StandardScaler())
])

categorical_pipeline = Pipeline([
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

# Combine into final preprocessor
preprocessor = ColumnTransformer([
    ('num', numeric_pipeline, num_v),
    ('cat', categorical_pipeline, cat_v)
])


# Question 4: Split data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("\nTraining set size:", X_train.shape[0])
print("Test set size:", X_test.shape[0])

# Clean the label and remove the spaces before or after it
y_train = y_train.str.strip()
y_test = y_test.str.strip()
# print("\nUnique labels in y_train:", y_train.unique())
# print("Unique labels in y_test:", y_test.unique())

# Question 5: Apply classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# Define five classifiers
classifiers = [
    ("Logistic Regression", LogisticRegression(max_iter=1000)),
    ("SVM", SVC()),
    ("Decision Tree", DecisionTreeClassifier()),
    ("Random Forest", RandomForestClassifier()),
    ("KNN", KNeighborsClassifier())
]

# Create an evaluation function
def evaluate_model(model, X_train, X_test, y_train, y_test):
    """Calculate and return the model evaluation indicators"""
    model.fit(X_train, y_train)
    y_predict = model.predict(X_test)
    
    # Make sure to use the cleaned labels
    pos_label = 'Approved' if 'Approved' in y_train.unique() else y_train.unique()[0]
    
    return {
        'accuracy': accuracy_score(y_test, y_predict),
        'precision': precision_score(y_test, y_predict, pos_label=pos_label),
        'recall': recall_score(y_test, y_predict, pos_label=pos_label),
        'f1': f1_score(y_test, y_predict, pos_label=pos_label),
        'report': classification_report(y_test, y_predict, target_names=['Rejected', 'Approved'])
    }

# Preprocess the data and evaluate all models
X_train_prep = preprocessor.fit_transform(X_train)
X_test_prep = preprocessor.transform(X_test)

for name, model in classifiers:
    metrics = evaluate_model(model, X_train_prep, X_test_prep, y_train, y_test)
    
    print(f"MODEL: {name}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    print("\nClassification Report:")
    print(metrics['report'])
    print("=="*50)

# Question 6: Ensemble Voting Classifier
print("\n" + "="*60)
print("ENSEMBLE VOTING CLASSIFIER (HARD VOTING)".center(50))
print("="*60)

# Create and evaluate voting classifier
voting = VotingClassifier(
    estimators=classifiers,
    voting='hard'
)

voting1 = evaluate_model(voting, X_train_prep, X_test_prep, y_train, y_test)

print(f"Hard voting")
print(f"{'Accuracy:'}{voting1['accuracy'] : .4f}")
print(f"{'Precision:'}{voting1['precision'] : .4f}")
print(f"{'Recall:'}{voting1['recall'] : .4f}")
print(f"{'F1 Score:'}{voting1['f1'] : .4f}")

print("\nClassification Report:")
print(voting1['report'])


print("The performance of Random Forest is the best among all models. " \
"\nSince it smoothes the anomaly judgment of one single tree by aggregating the predictions of multiple trees." \
"\nIt also can bootstrap, which enables each tree to focus on different data subsets." \
"\nthus avoiding the distortion of the global situation by certain dominant features.")