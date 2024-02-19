import pandas as pd
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import label_binarize
from itertools import cycle
from imblearn.over_sampling import SMOTE
from params import *

# Load the dataset
data = pd.read_csv('./data/top_20_features_data.csv')  # Ensure the correct path

X = data.drop('emotion', axis=1)
y = data['emotion']

if oversample == True:
    sm = SMOTE(random_state=42)
    X, y = sm.fit_resample(X, y)

# Perform 5 iterations of testing
num_iterations = 5
accuracy_scores = []
accuracy_scores2 = []
#model = LogisticRegression(max_iter=1000)
model = svm.SVC(probability=True)
X_train = X
y_train = y
for _ in range(num_iterations):
    # Split the dataset into training and testing subsets
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.3, random_state=100)
    print("lmao")
    # Train the model on the training data
    model.fit(X_train, y_train)
    # Make predictions on the testing data
    y_pred = model.predict(X_test)
    y_predt = model.predict(X_train)

    # Evaluate the accuracy of the model
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_scores.append(accuracy)

    accuracy2 = accuracy_score(y_train, y_predt)
    accuracy_scores2.append(accuracy2)
    

# Print the accuracy scores for each iteration
for i, acc in enumerate(accuracy_scores):
    print(f"Iteration {i+1}: Accuracy = {acc}")

# Calculate the average accuracy
average_accuracy = sum(accuracy_scores) / num_iterations
print(f"Average Accuracy: {average_accuracy}")

for i, acc2 in enumerate(accuracy_scores2):
    print(f"Iteration {i+1}: train Accuracy = {acc2}")

###### EVALUATION
emotions = {1:"happy", 2:"calm", 3:"angry", 4:"sad"}
## Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
#cm = pd.DataFrame(cm , index = [i for i in encoder.categories_] , columns = [i for i in encoder.categories_])
sns.heatmap(cm, linecolor='white', cmap='Blues', linewidth=1, annot=True, fmt='')
plt.title('Confusion Matrix', size=20)
plt.xlabel('Predicted Labels', size=4)
plt.ylabel('Actual Labels YTEST', size=4)
plt.show()

cm = confusion_matrix(y_train, y_predt)
#cm = pd.DataFrame(cm , index = [i for i in encoder.categories_] , columns = [i for i in encoder.categories_])
sns.heatmap(cm, linecolor='white', cmap='Blues', linewidth=1, annot=True, fmt='')
plt.title('Confusion Matrix', size=20)
plt.xlabel('Predicted Labels', size=4)
plt.ylabel('Actual Labels YTRAIN', size=4)


print(classification_report(y_test, y_pred))

n_classes = 4  # Number of classes
fpr = dict()
tpr = dict()
roc_auc = dict()
precision = dict()
recall = dict()
thresholds = dict()
y_probabilities = model.predict_proba(X_test)
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