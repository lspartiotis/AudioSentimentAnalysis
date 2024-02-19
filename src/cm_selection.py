import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from features import csv_to_pd
from params import *

np.random.seed(123)

no_deltas = False
if no_deltas == True:
    data = short_term
else:
    # this is the csv we extracted in previous step.
    # it contains all audio and valence&arousal features
    data = csv_to_pd(feature_path)
    data = data.drop(columns=['song_id'])
    print(data.info())

###STEP1 FEATURES VS FEATURES::::
#get correlation matrix
plt.figure()
corr = data.corr()
# Generate correlation heatmap
sns.heatmap(corr)
plt.figure()

# filter the values: because we have too many fearues (130x130) approximate matrix!
# we cannot conclude by looking at the heatmap.
# so, set a threshold for which we will consider the features
num_of_feats = corr.shape[0]
columns = np.full((corr.shape[0],), True, dtype=bool)
for i in range(corr.shape[0]):
    for j in range(i+1, corr.shape[0]):
        if abs(corr.iloc[i,j]) >= 0.9:
            if columns[j]:
                columns[j] = False
# Get selected cols
selected_columns = data.columns[columns]
data1 = data[selected_columns]
corr1 = data1.corr()
print(f'Dataset after { len(data.columns) - len(selected_columns) } dependent features droped: ', data1.head())
sns.heatmap(corr1)
plt.figure()

###STEP2 FEATURES VS LABELS::::
#get features with high correlation to the labels
corr_matrix = data.corr()
corr_matrix["emotion"].sort_values(ascending=False)

num_of_feats = corr_matrix.shape[0]
columns = np.full((corr_matrix.shape[0],), True, dtype=bool)
label_column = corr_matrix.iloc[:, -1]
for i in range(corr_matrix.shape[0]):
    if abs(label_column[i]) <= 0.3:
            columns[i] = False

# Get selected cols
selected_columnsf = data.columns[columns]
data2 = data[selected_columnsf]
corr2 = data2.corr()
print(f'Dataset after {len(data.columns) - len(selected_columnsf) } dependent features droped: ', data2.head())
sns.heatmap(corr2)
plt.figure()

#### PLOT THE DISTRIBUTION OF EACH SELECTED FEATURE
result = pd.DataFrame()
result['emotion'] = data.iloc[:,-1]
j = 1
subrows = (len(data2.columns) + 1) // 2
fig, axs = plt.subplots(subrows, 2, figsize=(16, 5*subrows))
plt.subplots_adjust(wspace=0.4, hspace=0.6)

for i in data2.columns[:-3]:
    plt.subplot(subrows, 2, j)
    sns.distplot(data2[i][result['emotion']==1], color='g', label = 'happy', bins=30, kde_kws={'bw': 0.5})
    sns.distplot(data2[i][result['emotion']==2], color='r', label = 'calm', bins=30, kde_kws={'bw': 0.5})
    sns.distplot(data2[i][result['emotion']==3], color='b', label = 'angry', bins=30, kde_kws={'bw': 0.5})
    sns.distplot(data2[i][result['emotion']==4], color='m', label = 'sad', bins=30, kde_kws={'bw': 0.5})
    plt.legend(loc='best')
    j += 1

# Selecting useful features from training and test data
# if datta=1, take the filttered features from features vs features
# else from features vs labels
datta = 2
if datta == 1:
    X_filtered = np.array(data1.iloc[:, :-3].values)#.T
    y_filtered = np.array(data1['emotion'])
    y_full_filtered = np.array([data1['valence'],data1['arousal']])
else:
    X_filtered = np.array(data2.iloc[:, :-3].values)#.T
    y_filtered = np.array(data2['emotion'])
    y_full_filtered = np.array([data2['valence'],data2['arousal']])
print('Selecting features...')
print(y_filtered.shape, X_filtered.shape)

plt.show()