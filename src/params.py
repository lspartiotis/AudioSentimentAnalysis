from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

feature_path = './data/total_annotated_data.csv'
oversample = True
classifier = RandomForestClassifier(random_state=42)
#SVC(probability=True)