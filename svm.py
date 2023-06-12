import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, auc
from sklearn.decomposition import PCA

np.random.seed(42)



def run_svm():

  # X_values = pd.read_csv('https://raw.githubusercontent.com/iamtatha/Mental-Health/main/Data/all_label_train.csv')
  # Y_values = pd.read_csv('https://raw.githubusercontent.com/iamtatha/Mental-Health/main/Data/all_label_target.csv')

  # X_values = pd.read_csv('https://raw.githubusercontent.com/iamtatha/Mental-Health/main/Data/wo_onehot_train.csv')
  # Y_values = pd.read_csv('https://raw.githubusercontent.com/iamtatha/Mental-Health/main/Data/wo_onehot_target.csv')

  X_values = pd.read_csv('https://raw.githubusercontent.com/iamtatha/Mental-Health/main/Data/train.csv')
  Y_values = pd.read_csv('https://raw.githubusercontent.com/iamtatha/Mental-Health/main/Data/target.csv')


  df = pd.concat([X_values, Y_values],axis=1)

  scaled_data = df
  scaler = MinMaxScaler(feature_range=(0,1))
  scaled_data = pd.DataFrame(scaler.fit_transform(scaled_data))

  X_train, X_test, Y_train, Y_test = train_test_split(scaled_data.iloc[:,:-1], scaled_data.iloc[:,-1], train_size=0.80)

  pca = PCA(n_components = 'mle')
  X_train = pca.fit_transform(X_train)
  X_test = pca.transform(X_test)
  n = np.array(X_train).shape[1]

  classifier = SVC(kernel='rbf', random_state = 1, gamma='auto', probability=True)
  # classifier = SVC()
  metric = classifier.fit(X_train, Y_train)
  pred = classifier.predict(X_test)
  print(accuracy_score(pred, Y_test))
  print("Precision Score: ", precision_score(Y_test, pred))
  print("Recall Score: ", recall_score(Y_test, pred))
  print("F1 Score: ", f1_score(Y_test, pred))
  return accuracy_score(pred, Y_test), precision_score(Y_test, pred), recall_score(Y_test, pred), f1_score(Y_test, pred), accuracy_score(classifier.predict(X_train), Y_train)


run_svm()




