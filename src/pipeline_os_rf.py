import pandas as pd
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import label_binarize
import seaborn as sns
from itertools import cycle
from sklearn.svm import SVC
from params import *
# Load the dataset
data = pd.read_csv('./data/top_20_features_data.csv')

# Separate the features and target variable
X = data.drop('emotion', axis=1)
y = data['emotion']

# Define the parameter grid for Random Forest
param_gridrf = {
    'classifier__n_estimators': [50, 100, 200],
    'classifier__max_depth': [None, 10, 20, 30],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__min_samples_leaf': [1, 2, 4],
    'classifier__bootstrap': [True, False]
}

param_grid_svc = {
    'C': [0.1, 1, 10],
    'gamma': [0.001, 0.01, 0.1],
    'kernel': ['linear', 'rbf', 'poly']
}

# Initialize a Random Forest classifier
#rf = SVC(probability=True)
rf = RandomForestClassifier(random_state=42)

# Create a pipeline that first applies SMOTE and then fits the Random Forest classifier
pipeline = Pipeline([
    ('oversample', SMOTE(random_state=42)),
    ('classifier', rf)
])


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y, shuffle=True)
if oversample == True:
    sm = SMOTE(random_state=42)
    X_train, y_train = sm.fit_resample(X_train, y_train)
# Define the KFold cross-validator
kf = KFold(n_splits=5, shuffle=True, random_state=42) 
# Initialize the Grid Search model with cross-validation
# estimator = pipeline
grid_search = GridSearchCV(estimator=pipeline, param_grid=param_gridrf, cv=kf, scoring='accuracy', n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
best_score = grid_search.best_score_

best_model = grid_search.best_estimator_

y_test_pred = best_model.predict(X_test)
y_train_pred = best_model.predict(X_train)
# Obtain the predicted probabilities for the test set
y_probabilities = best_model.predict_proba(X_test)

validation_accuracy = best_model.score(X_test, y_test)
training_accuracy = best_model.score(X_train, y_train)
print("Best Parameters:", best_params)
print("Best Score:", best_score)
print("Test Accuracy:", validation_accuracy)
print("Training Accuracy:", training_accuracy)
print("Classification Report:")
print(classification_report(y_test, y_test_pred))

###### EVALUATION
emotions = {1:"happy", 2:"calm", 3:"angry", 4:"sad"}
## Confusion Matrix
cm = confusion_matrix(y_test, y_test_pred)
#cm = pd.DataFrame(cm , index = [i for i in encoder.categories_] , columns = [i for i in encoder.categories_])
sns.heatmap(cm, linecolor='white', cmap='Blues', linewidth=1, annot=True, fmt='')
plt.title('Confusion Matrix', size=20)
plt.xlabel('Predicted Labels', size=4)
plt.ylabel('Actual Labels YTEST', size=4)
plt.show()

cm = confusion_matrix(y_train, y_train_pred)
#cm = pd.DataFrame(cm , index = [i for i in encoder.categories_] , columns = [i for i in encoder.categories_])
sns.heatmap(cm, linecolor='white', cmap='Blues', linewidth=1, annot=True, fmt='')
plt.title('Confusion Matrix', size=20)
plt.xlabel('Predicted Labels', size=4)
plt.ylabel('Actual Labels YTRAIN', size=4)
plt.show()

print(classification_report(y_test, y_test_pred))

n_classes = 4  # Number of classes
fpr = dict()
tpr = dict()
roc_auc = dict()
precision = dict()
recall = dict()
thresholds = dict()
y_probabilities = best_model.predict_proba(X_test)
y_test_binary = label_binarize(y_test,classes=[1, 2, 3, 4])  # Binarize the true labels
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_binary[:, i], y_probabilities[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    precision[i], recall[i], thresholds[i] = precision_recall_curve(y_test_binary[:, i], y_probabilities[:, i])

# Plot ROC curves for each class
plt.figure(figsize=(8, 6))
colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label='ROC curve of class {0} (AUC = {1:0.2f})'
             ''.format(i+1, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC-AUC Curves for Multiclass Classification')
plt.legend(loc="lower right")
plt.show()

# Plot precision recal
plt.figure(figsize=(8, 6))
colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green'])
for i, color in zip(range(n_classes), colors):
    plt.plot(recall[i], precision[i], color=color, lw=2,
             label='Precision-Recall curve of class {0}'.format(i+1))

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curves for Multiclass Classification')
plt.legend(loc="lower left")
plt.show()
