import pandas as pd
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Load the dataset
data = pd.read_csv('./data/top_20_features_data.csv')

# Separate the features and target variable
X = data.drop('emotion', axis=1)
y = data['emotion']

# Initialize classifiers
classifiers = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100),
    'Support Vector Machine': SVC(probability=True)
}

# Define the KFold cross-validator
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Perform k-fold cross-validation for each classifier
for name, clf in classifiers.items():
    cv_scores = cross_val_score(clf, X, y, cv=kf, scoring='accuracy')
    
    print(f"{name} Average Cross-Validation Accuracy: {cv_scores.mean():.4f}")
    print(f"{name} Cross-Validation Scores: {cv_scores}")
    print(f"{name} Standard Deviation of CV Scores: {cv_scores.std():.4f}\n")
