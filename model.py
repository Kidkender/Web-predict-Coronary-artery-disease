import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import pickle
from sklearn.metrics import accuracy_score


data = pd.read_csv('coronary_prediction.csv')
data = data.dropna(subset=['education','cigsPerDay','BPMeds','totChol','BMI','heartRate','glucose'])
data = data.iloc[:1200, :]

X = data.drop('TenYearCHD', axis=1)
y = data['TenYearCHD']

smote = SMOTE()
X, y = smote.fit_resample(X, y)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lr = make_pipeline(StandardScaler(), LogisticRegression())
knn = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))
rf = RandomForestClassifier(n_estimators=10, random_state=42)
estimators = [('lr', lr), ('knn', knn), ('rf', rf)]
clf = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())

clf.fit(X_train, y_train)
y_pred = clf.predict_proba(X_test)[:, 1]



#Save the model

# with open('best_model_.pkl', 'wb') as f:
#     pickle.dump(clf, f)

#_______________________________________
#TESTING THE MODEL
feature_names = ['male', 'age', 'education', 'currentSmoker', 'cigsPerDay', 'BPMeds', 'prevalentStroke', 'prevalentHyp', 'diabetes', 'totChol', 'sysBP'
                ,'diaBP', 'BMI', 'heartRate', 'glucose']

array_features_NONCAD = [np.array([0,43,2,0,0,0,0,0,0,247,131,88,27.64,72,61])]
# Create DataFrame from array with column names
input_data_NONCAD = pd.DataFrame(data=array_features_NONCAD, columns=feature_names)
print("Data Frame", input_data_NONCAD)

array_features_CAD = [np.array([0,41,3,1,30,0,0,1,0,187,154,100,20.5,66,78])]

# Create DataFrame from array with column names
input_data_CAD = pd.DataFrame(data=array_features_CAD, columns=feature_names)
print("Data Frame", input_data_CAD)

model = pickle.load(open('best_model_.pkl', 'rb'))
print("Prediction on Data with Non - CAD Target:", model.predict(input_data_NONCAD))
print("Prediction on Data with CAD Target:", model.predict(input_data_CAD))
#_______________________________________
