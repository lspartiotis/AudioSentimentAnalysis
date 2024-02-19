import pandas as pd
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the dataset
data = pd.read_csv('./data/top_20_features_data.csv')  # Ensure the correct path

# Separate the features and target variable
X = data.drop('emotion', axis=1)
y = data['emotion']

# Split the data into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the parameter grid for Random Forest
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

# Initialize a Random Forest classifier
rf = RandomForestClassifier(random_state=42)

# Define the KFold cross-validator
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Initialize the Grid Search model with cross-validation
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=kf, scoring='accuracy', n_jobs=-1, verbose=1)

# Perform the Grid Search with cross-validation on the training data
grid_search.fit(X_train, y_train)

# Print the best parameters and the best cross-validation score
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best cross-validation accuracy (Training): {grid_search.best_score_:.4f}")

# Evaluate the best estimator on the held-out test set
best_estimator = grid_search.best_estimator_
y_pred_train = best_estimator.predict(X_train)
y_pred_test = best_estimator.predict(X_test)

print(f"Training Accuracy: {accuracy_score(y_train, y_pred_train):.4f}")
print(f"Test Accuracy: {accuracy_score(y_test, y_pred_test):.4f}")
