### Visualizing different labels count
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from features import csv_to_pd



feature_path = './data/total_annotated_data.csv'

data = csv_to_pd(feature_path)
X = np.array(data.iloc[:, :-3].values)#.T
y = np.array(data['emotion'])
y_r = np.array([data['valence'],data['arousal']])
print(y_r)
feature_names = data.columns.values
print(X.shape)
print(y.shape)
print(feature_names)


plt.title('Count of Emotions', size=4)
sns.countplot(x='emotion', data=data, palette=['green', 'blue', 'orange', 'purple'])
plt.ylabel('Count')
plt.xlabel('Emotions')
sns.despine(top=True, right=True, left=False, bottom=False)
plt.show()



## dataset without the std values
filter_columns = list(range(0, 67)) + list(range(136, 141))
print(data.shape)
midterm_mean = data.iloc[:, filter_columns].copy()
## dataset without the deltas and std values (short term features)
filter_columns = list(range(0, 34)) + list(range(136, 141))
short_term = data.iloc[:, filter_columns].copy()

midterm_mean.to_csv("./data/total_data_no_std.csv",index=False)
short_term.to_csv("./data/total_data_no_std_no_deltas.csv",index=False)

X_means = np.array(short_term.iloc[:, :-3].values)#.T
y_means = np.array(data['emotion'])
y_means_r = np.array([short_term['valence'],short_term['arousal']])

no_std = 2
if no_std == 2:
    X = X_means
    y = y_means
    y_r = y_means_r

print(X.shape)

print(short_term.iloc[:, :-3])

subrows = (len(data.columns) + 1) // 2
fig, axs = plt.subplots(subrows, 2, figsize=(16, 5*subrows))
plt.subplots_adjust(wspace=0.4, hspace=0.6)

result = pd.DataFrame
result= data.iloc[:, -1]
j = 1
for i in data.columns[:-3]:
    plt.subplot(subrows, 2, j)
    sns.distplot(data[i][result['emotion']==1], color='g', label = 'happy', bins=30, kde_kws={'bw': 0.5})
    sns.distplot(data[i][result['emotion']==2], color='r', label = 'calm', bins=30, kde_kws={'bw': 0.5})
    sns.distplot(data[i][result['emotion']==3], color='b', label = 'angry', bins=30, kde_kws={'bw': 0.5})
    sns.distplot(data[i][result['emotion']==4], color='m', label = 'sad', bins=30, kde_kws={'bw': 0.5})
    plt.legend(loc='best')
    j += 1

plt.show()