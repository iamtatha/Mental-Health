import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.decomposition import PCA


# X_values = pd.read_csv('https://raw.githubusercontent.com/iamtatha/Mental-Health/main/Data/all_label_train.csv')
# Y_values = pd.read_csv('https://raw.githubusercontent.com/iamtatha/Mental-Health/main/Data/all_label_target.csv')

X_values = pd.read_csv('https://raw.githubusercontent.com/iamtatha/Mental-Health/main/Data/train.csv')
Y_values = pd.read_csv('https://raw.githubusercontent.com/iamtatha/Mental-Health/main/Data/target.csv')


df = pd.concat([X_values, Y_values],axis=1)

scaled_data = df
# scaler = MinMaxScaler(feature_range=(0,1))
# scaled_data = pd.DataFrame(scaler.fit_transform(scaled_data))

X_train, X_test, Y_train, Y_test = train_test_split(scaled_data.iloc[:,:-1], scaled_data.iloc[:,-1], train_size=0.90)
n = np.array(X_train).shape[1]


pca = PCA(n_components = 'mle')
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
n = np.array(X_train).shape[1]
print(n)


model = Sequential()
model.add(Dense(20, input_dim = n, activation = 'relu'))
model.add(Dense(10, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))

model.compile(loss='categorical_crossentropy', optimizer='RMSprop', metrics=['accuracy', 'mse'])


learnt = model.fit(X_train, Y_train, epochs=100, batch_size=64)



Y_pred = model.predict(X_test)
X = []
for i in range(len(list(Y_pred))):
  X.append(i+1)
  
  
  
print(accuracy_score(Y_test, Y_pred))
print(precision_score(Y_test, Y_pred))
print(recall_score(Y_test, Y_pred))
print(f1_score(Y_test, Y_pred))
print(confusion_matrix(Y_test, Y_pred))
