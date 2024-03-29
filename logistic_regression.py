# -*- coding: utf-8 -*-
"""logistic_regression.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1jxWgB4Qm4IkgZ76FwFgBAbGNgOcvpgYP
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import keras
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression

import sys
import numpy
numpy.set_printoptions(threshold=sys.maxsize)

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

#Without one-hot encoding.
np.random.seed(43)

#without one-hot encoding
X_values = pd.read_csv('https://raw.githubusercontent.com/iamtatha/Mental-Health/main/Data/all_label_train.csv')
y_values = pd.read_csv('https://raw.githubusercontent.com/iamtatha/Mental-Health/main/Data/all_label_target.csv')
X_data = pd.DataFrame(X_values)
y_data = pd.DataFrame(y_values)
#data set division. (here i have this error
df = pd.concat([X_data, y_data],axis=1)
#datadiv(df)
X_train,X_test,y_train,ytest = train_test_split(df.iloc[:,:-1],df.iloc[:,-1],train_size=0.75)

#model without optimization
logistic_regModel = LogisticRegression(max_iter = 800)
logistic_regModel.fit(X_train,y_train) 
#training accuracy
print("Train accuracy without optimization + without one-hot encoding: ",logistic_regModel.score(X_train,y_train))
#test accuracy
print("Test accuracy without optimization + without one-hot encoding: ",logistic_regModel.score(X_test,ytest))
predictions = logistic_regModel.predict(X_test)
#confusion matrix
print("Confusion matrix without optimization + without one-hot encoding: ",confusion_matrix(ytest,predictions))
#precision
print("Precision without optimization + without one-hot encoding: ",precision_score(ytest,predictions))
#recall
print("Recall without optimization + without one-hot encoding: ",recall_score(ytest,predictions))
#f1-score
print("F1 Score without optimization + without one-hot encoding: ",f1_score(ytest,predictions))

#modified lr model
lr_model = LogisticRegression(penalty='l1', solver='liblinear')
lr_model.fit(X_train,y_train) 
#train accuracy
print("Train accuracy without one hot encoding + with optimization: ",lr_model.score(X_train,y_train))
lr_model.score(X_train,y_train)
#test accuracy
print("Test accuracy without one-hot encoding + with optimization: ", lr_model.score(X_test,ytest))
predictions = lr_model.predict(X_test)
#confusion matrix
print("Confusion matrix without one-hot encoding + with optimization: ",confusion_matrix(ytest,predictions))
#precision
print("Precision with optimization + without one-hot encoding: ",precision_score(ytest,predictions))
#recall
print("Recall with optimization + without one-hot encoding: ",recall_score(ytest,predictions))
#f1-score
print("F1 Score with optimization + without one-hot encoding: ",f1_score(ytest,predictions))

#with one-hot encoding
np.random.seed(43)

X_values = pd.read_csv('https://raw.githubusercontent.com/iamtatha/Mental-Health/main/Data/train.csv')
y_values = pd.read_csv('https://raw.githubusercontent.com/iamtatha/Mental-Health/main/Data/target.csv')
X_data = pd.DataFrame(X_values)
y_data = pd.DataFrame(y_values)
dd = pd.concat([X_data, y_data],axis=1)
#datadiv(dd)
X_train,X_test,y_train,ytest = train_test_split(dd.iloc[:,:-1],dd.iloc[:,-1],train_size=0.75)

#model without optimization
logistic_regModel = LogisticRegression(max_iter = 800)
logistic_regModel.fit(X_train,y_train) 
#training accuracy
print("Train accuracy without optimization + one hot encoding: ",logistic_regModel.score(X_train,y_train))
#test accuracy
print("Test accuracy without optimization + one hot encoding: ",logistic_regModel.score(X_test,ytest))
predictions = logistic_regModel.predict(X_test)
#confusion matrix
print("confusion matrix without optimization + one hot encoding: ",confusion_matrix(ytest,predictions))
#precision
print("Precision without optimization + with one-hot encoding: ",precision_score(ytest,predictions))
#recall
print("Recall without optimization + with one-hot encoding: ",recall_score(ytest,predictions))
#f1-score
print("F1 Score without optimization + with one-hot encoding: ",f1_score(ytest,predictions))

#modified lr model with one hot encoding
lr_model = LogisticRegression( penalty='l1', solver='liblinear')
lr_model.fit(X_train,y_train) 
#train accuracy
print("Train accuracy with one-hot encoding + with optimization:",lr_model.score(X_train,y_train))
#test accuracy
print("Test accuracy one-hot encoding + with optimization:", lr_model.score(X_test,ytest))
predictions = lr_model.predict(X_test)
#confusion matrix
print("Confusion matrix one-hot encoding + with optimization: ",confusion_matrix(ytest,predictions))
#precision
print("Precision with optimization + with one-hot encoding: ",precision_score(ytest,predictions))
#recall
print("Recall with optimization + with one-hot encoding: ",recall_score(ytest,predictions))
#f1-score
print("F1 Score with optimization + with one-hot encoding: ",f1_score(ytest,predictions))
