
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, classification_report

emotions = {1:"happy", 2:"calm", 3:"angry", 4:"sad"}
## Confusion Matrix
cm = confusion_matrix(y_test, y_predt)
#cm = pd.DataFrame(cm , index = [i for i in encoder.categories_] , columns = [i for i in encoder.categories_])
sns.heatmap(cm, linecolor='white', cmap='Blues', linewidth=1, annot=True, fmt='')
plt.title('Confusion Matrix', size=20)
plt.xlabel('Predicted Labels', size=4)
plt.ylabel('Actual Labels YTEST', size=4)
plt.show()

cm = confusion_matrix(y_train, y_pred)
#cm = pd.DataFrame(cm , index = [i for i in encoder.categories_] , columns = [i for i in encoder.categories_])
sns.heatmap(cm, linecolor='white', cmap='Blues', linewidth=1, annot=True, fmt='')
plt.title('Confusion Matrix', size=20)
plt.xlabel('Predicted Labels', size=4)
plt.ylabel('Actual Labels YTRAIN', size=4)

