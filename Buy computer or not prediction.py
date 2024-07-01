import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

dataset = pd.read_csv('/content/computer.csv')

X = dataset.iloc[:, [2, 3, 4]].values

y = dataset.iloc[:, -1].values

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20,
random_state=1)

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)

from sklearn.naive_bayes import GaussianNB

classifier = GaussianNB()

classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

print("Predicted values:")

print(y_pred)

print("Actual values:")

print(y_test)

from sklearn.metrics import confusion_matrix, accuracy_score

ac = accuracy_score(y_test, y_pred)

cm = confusion_matrix(y_test, y_pred)

print("Accuracy:"
, ac)

print("Confusion Matrix:")

print(cm)
