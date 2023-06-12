# -*- coding: utf-8 -*-
"""KNN.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1QFYLdQDEMplq4ldotWi_NR79d_7oXl_j
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import StackingClassifier
import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestRegressor, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA

import math
math.sqrt(len(Y_test))

def KNN():
  X_values = pd.read_csv('https://raw.githubusercontent.com/iamtatha/Mental-Health/main/Data/train.csv')
  Y_values = pd.read_csv('https://raw.githubusercontent.com/iamtatha/Mental-Health/main/Data/target.csv')
    

  df = pd.concat([X_values, Y_values],axis=1)

  scaled_data = df
  scaler = MinMaxScaler(feature_range=(0,1))
  scaled_data = pd.DataFrame(scaler.fit_transform(scaled_data))

  X_train, X_test, Y_train, Y_test = train_test_split(scaled_data.iloc[:,:-1], scaled_data.iloc[:,-1], train_size=0.70)

  from sklearn.neighbors import KNeighborsClassifier
  knnclassifier = KNeighborsClassifier(n_neighbors = 19)
  knnclassifier.fit(X_train, Y_train)

  y_predtrain = knnclassifier.predict(X_train)
  y_predtest = knnclassifier.predict(X_test)
  print("KNN Classifier Accuracy: ", accuracy_score(Y_train, y_predtrain))
  print("KNN Classifier Accuracy: ", accuracy_score(Y_test, y_predtest))

KNN()
