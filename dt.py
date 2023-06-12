import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def import_():
	data = pd.read_csv('https://raw.githubusercontent.com/iamtatha/Mental-Health/main/Data/all_label_train.csv')
	target = pd.read_csv('https://raw.githubusercontent.com/iamtatha/Mental-Health/main/Data/all_label_target.csv')

	return data, target

def scale(data):
  scaler = MinMaxScaler(feature_range=(0,1))
  scaler = StandardScaler()
  scaled_data = pd.DataFrame(scaler.fit_transform(data))
  return data


def split_(data, target):
	X_train, X_test, y_train, y_test = train_test_split(data, target, test_size = 0.35, random_state = 100)
	return data, target, X_train, X_test, y_train, y_test

def pca_analysis(X_train, X_test):
  pca = PCA(n_components = 'mle')
  X_train = pca.fit_transform(X_train)
  X_test = pca.transform(X_test)
  return X_train, X_test

def train(X_train, X_test, y_train):
	model_dt = DecisionTreeClassifier(criterion = "entropy", splitter="best" , random_state = 20, max_depth = 3) #change hp

	model_dt.fit(X_train, y_train)
	return model_dt

def pred(X_test, model_dt):
	y_pred = model_dt.predict(X_test)
	return y_pred
	
def accuracy(y_test, y_pred):
    print("Confusion Matrix: ")
    print(confusion_matrix(y_test, y_pred))
    print ("Accuracy : ",accuracy_score(y_test,y_pred)*100)
    print("Report : ",classification_report(y_test, y_pred))

def main():
  data, target = import_()
  data = scale(data)
  data, target, X_train, X_test, y_train, y_test = split_(data, target)
  X_train, X_test = pca_analysis(X_train, X_test)
  model_dt = train(X_train, X_test, y_train)

  y_pred = pred(X_test, model_dt)
  accuracy(y_test, y_pred)
	
if __name__=="__main__":
	main()
