import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import matplotlib.pyplot as plt

# Load the annotated dataset
data = pd.read_csv('./data/total_annotated_data.csv')

# Preprocess the dataset by dropping irrelevant columns
X = data.drop(['Unnamed: 0', 'valence', 'arousal', 'emotion', 'song_id'], axis=1)
y = data['emotion']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest classifier to get feature importances
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Extracting feature importance and sorting features
features = X.columns
importances = rf.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': features, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Initialize variables to track the best performance
best_accuracy = 0
best_n_features = 0
accuracies = []

# Iteratively evaluate model performance with an increasing number of top features
for n_features in range(1, len(features) + 1):
    top_n_features = feature_importance_df['Feature'].head(n_features).values
    
    # Selecting the top N features for training and testing
    X_train_selected = X_train[top_n_features]
    X_test_selected = X_test[top_n_features]
    
    # Reinitialize the model to train with the top N features
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_selected, y_train)
    
    # Use cross-validation to evaluate the model
    scores = cross_val_score(model, X_test_selected, y_test, cv=5, scoring='accuracy')
    current_accuracy = np.mean(scores)
    
    # Update best performance metrics
    if current_accuracy > best_accuracy:
        best_accuracy = current_accuracy
        best_n_features = n_features
    
    # Store accuracy for potential plotting
    accuracies.append(current_accuracy)

# After finding the best number of features, retrain to get the feature importances
top_features = feature_importance_df['Feature'].head(best_n_features).values

# Now use the entire dataset to filter only those top features
X_top_features = data[top_features]

# Add the target variable 'emotion' back into the DataFrame
X_top_features['emotion'] = data['emotion']

# Save this new DataFrame with only the top N features to a CSV file
output_csv_path = './data/top_features_based_on_accuracy.csv'
X_top_features.to_csv(output_csv_path, index=False)

print(f'File saved to {output_csv_path}')

