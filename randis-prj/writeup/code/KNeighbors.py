import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

#random seed to reproduce results
np.random.seed(42)

#read in csv and massage the training data 
train = pd.read_csv('train.csv')
x_train = train.drop(['id', 'species'], axis=1).values
le = LabelEncoder().fit(train['species']) 					#encode species strings
y_train = le.transform(train['species'])
scaler = StandardScaler().fit(x_train)
x_train = scaler.transform(x_train)							#standardize features

#Run KNeighbors Classifier to train the model
clf = KNeighborsClassifier(3)
clf.fit(x_train, y_train)

#read in and massage the test data
test = pd.read_csv('test.csv')
test_ids = test.pop('id')
x_test = test.values
x_test = scaler.transform(x_test)

# Predict Test Set
y_test = clf.predict_proba(x_test)

#format and output for csv submission file
submission = pd.DataFrame(y_test, index=test_ids, columns=le.classes_)
submission.to_csv('submission_KNeighbors.csv')
print("Success! Check your directory for the submission_KNeighbors file, you handsome data scientist you.")