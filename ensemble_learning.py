import random

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import keras
from sklearn import preprocessing
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score,f1_score,precision_score, recall_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestRegressor, AdaBoostClassifier, GradientBoostingClassifier, \
    RandomForestClassifier, StackingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import sklearn.metrics
import six
import sys
sys.modules['sklearn.externals.six'] = six
import mlrose

import tensorflow as tf

def cv_score_cal(model, x_data, y_data, cv_fold,model_name):
    result = cross_val_score(model, x_data,y_data,cv=cv_fold)
    print(model_name + " accuracy: " + str(result.mean()), ", std_dev: " + str(result.std()))


def recall_precision(y_true, y_pred, model_name):
    print(model_name, " recall : ", recall_score(ytest,y_pred))
    print(model_name, " precision : ", precision_score(ytest, y_pred))


import sys
import numpy
numpy.set_printoptions(threshold=sys.maxsize)

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# X_values = pd.read_csv('train.csv')
# y_values = pd.read_csv('target.csv')

X_values = pd.read_csv('https://raw.githubusercontent.com/iamtatha/Mental-Health/main/Data/all_label_train.csv')
y_values = pd.read_csv('https://raw.githubusercontent.com/iamtatha/Mental-Health/main/Data/all_label_target.csv')
# X_values = pd.read_csv('https://raw.githubusercontent.com/iamtatha/Mental-Health/main/Data/train.csv')
# y_values = pd.read_csv('https://raw.githubusercontent.com/iamtatha/Mental-Health/main/Data/target.csv')


X_data = pd.DataFrame(X_values)
y_data = pd.DataFrame(y_values)

df = pd.concat([X_data, y_data],axis=1)

random.seed(10000)

#data set division. (here i have this error

# scaled_data = df
# scaler = MinMaxScaler(feature_range=(0,1))
# scaled_data = pd.DataFrame(scaler.fit_transform(scaled_data))

# X_total = scaled_data.iloc[:,:-1]
# y_total = scaled_data.iloc[:,-1]

X_total = df.iloc[:,:-1]
y_total = df.iloc[:,-1]

# X_train,X_test,y_train,ytest = train_test_split(scaled_data.iloc[:,:-1],scaled_data.iloc[:,-1],train_size=0.80,random_state=100)
X_train,X_test,y_train,ytest = train_test_split(df.iloc[:,:-1],df.iloc[:,-1],train_size=0.70,random_state=100)
print(y_train.sum())
print(len(X_train))

random_forest_classifier = RandomForestClassifier(n_estimators = 300, random_state = 0)
random_forest_classifier.fit(X_train,y_train)
random_forest_classifier.score(X_test,ytest)
# predictions_train = random_forest_regressor.predict(X_train)
# predictions_test = random_forest_regressor.predict(X_test)
# print("RandomForest Train Accuracy: ", accuracy_score(y_train, predictions_train>0.5))
# print("RandomForest Test Accuracy: ", accuracy_score(ytest,predictions_test>0.5))

print("RandomForest Train Accuracy: ", accuracy_score(y_train, random_forest_classifier.predict(X_train)))
print("RandomForest Test Accuracy: ", accuracy_score(ytest,random_forest_classifier.predict(X_test)))
print("RandomForest F1 score: ",f1_score(ytest,random_forest_classifier.predict(X_test) , average='macro'))
cv_score_cal(random_forest_classifier, X_train, y_train, cv_fold =5, model_name="Random Forest Cross Validation")
recall_precision(ytest, random_forest_classifier.predict(X_test),"random Forest")
print(confusion_matrix(y_true=ytest, y_pred=random_forest_classifier.predict(X_test)))
#
#
#Adaboost
dc_classifier = DecisionTreeClassifier(max_leaf_nodes=2, random_state=100)
ada_boot_classifier = AdaBoostClassifier(n_estimators = 60)
ada_boot_classifier.fit(X_train, y_train)
print("Adaboost Train Accuracy: ", ada_boot_classifier.score(X_train, y_train))
print("Adaboost Test Accuracy: ",ada_boot_classifier.score(X_test,ytest))
y_pred = ada_boot_classifier.predict(X_test)
print("Adaboost F1 score: ",f1_score(ytest, y_pred, average='macro'))
cv_score_cal(ada_boot_classifier, X_total, y_total, cv_fold =5, model_name="Adaboost Cross Validation")
recall_precision(ytest, ada_boot_classifier.predict(X_test),"Adaboost ")
print(confusion_matrix(y_true=ytest, y_pred=ada_boot_classifier.predict(X_test)))

#
# #GradientBoost
gradient_boost_classifier = GradientBoostingClassifier(subsample=0.8,random_state=40)
gradient_boost_classifier.fit(X_train, y_train)
print("Gradient Train Accuracy: ", gradient_boost_classifier.score(X_train, y_train))
print("Gradient Test Accuracy: ",gradient_boost_classifier.score(X_test,ytest))
y_pred = gradient_boost_classifier.predict(X_test)
print("Gradient F1 score: ",f1_score(ytest, y_pred, average='macro'))
cv_score_cal(gradient_boost_classifier, X_total, y_total, cv_fold =5, model_name="Gradient Boost Cross Validation")
recall_precision(ytest, gradient_boost_classifier.predict(X_test),"Gradient boost")
print(confusion_matrix(y_true=ytest, y_pred=gradient_boost_classifier.predict(X_test)))

#
# #XGBoost
xg_boost_classifier = XGBClassifier()
xg_boost_classifier.fit(X_train, y_train)
print("XGBoost Train Accuracy: ", xg_boost_classifier.score(X_train, y_train))
print("XGBoost Test Accuracy: ",xg_boost_classifier.score(X_test,ytest))
y_pred = xg_boost_classifier.predict(X_test)
print("XGBoost F1 score: ",f1_score(ytest, y_pred, average='macro'))
cv_score_cal(xg_boost_classifier, X_total, y_total, cv_fold =5, model_name="XGBoost Cross Validation")
recall_precision(ytest, xg_boost_classifier.predict(X_test),"XGBoost ")
#
#

## Experiments with various models
# #Logistic Regression
# logistic_regression = LogisticRegression()
# logistic_regression.fit(X_train, y_train)
# print("Logistic Train Accuracy: ", logistic_regression.score(X_train, y_train))
# print("Logistic Test Accuracy: ",logistic_regression.score(X_test,ytest))
# y_pred = logistic_regression.predict(X_test)
# print("Logistic F1 score: ",f1_score(ytest, y_pred, average='macro'))
# #As we can see that ensemble learning is proving to give better results than rest of the classifiers we shifted towards them for hyper parameter tuning
#
#
#
# #KNN
# KNN_classifier = KNeighborsClassifier(n_neighbors=35)
# KNN_classifier.fit(X_train, y_train)
# print("KNN Train Accuracy: ", logistic_regression.score(X_train, y_train))
# print("KNN Test Accuracy: ",logistic_regression.score(X_test,ytest))
# y_pred = KNN_classifier.predict(X_test)
# print("KNN F1 score: ",f1_score(ytest, y_pred, average='macro'))
#
#SVM
svm_classifier = SVC()
metric = svm_classifier.fit(X_train, y_train)
print("SVM Train Accuracy: ", svm_classifier.score(X_train, y_train))
print("SVM Test Accuracy: ",svm_classifier.score(X_test,ytest))
y_pred = svm_classifier.predict(X_test)
print("SVM F1 score: ",f1_score(ytest, y_pred, average='macro'))
cv_score_cal(svm_classifier, X_total, y_total, cv_fold =5, model_name="SVM Cross Validation")
recall_precision(ytest, svm_classifier.predict(X_test),"SVM Classifier ")


#
# #Decision Tree
# decision_tree_classifier = DecisionTreeClassifier(criterion="entropy", splitter="best", random_state=20, max_depth=2)  # change hp
# decision_tree_classifier.fit(X_train, y_train)
# print("Decision Tree Train Accuracy: ", decision_tree_classifier.score(X_train, y_train))
# print("Decision Tree Test Accuracy: ",decision_tree_classifier.score(X_test,ytest))
# y_pred = decision_tree_classifier.predict(X_test)
# print("Decision Tree F1 score: ",f1_score(ytest, y_pred, average='macro'))
# cv_score_cal(decision_tree_classifier, X_train, y_train, 5 , "Decision Tree")
#
#
# #Stacking
# # clf1 = KNeighborsClassifier(n_neighbors=1)
# # clf2 = RandomForestClassifier(random_state=1)
# # clf3 = GaussianNB()
# # lr = LogisticRegression()
# # stack = StackingClassifier(classifiers=[clf1, clf2, clf3], meta_classifier=lr)
# # stack.fit(X_train, y_train)
# # # make class predictions for the testing set
# # y_pred_class = stack.predict(X_test)
# # print("Stacking accuracy : ", accuracy_score(ytest, y_pred_class))
# # print("Stacking F1 score: ",f1_score(ytest, y_pred_class, average='macro'))
#
#
# param_grid = [
#     {
#         'n_estimators': [5,10,70,50,100], 'max_features' : [2,4,6,10,12],
#         'max_depth' : [2,3,4,5]
#     }
# ]
# grid_random_forest = RandomForestClassifier()
# grid_result = GridSearchCV(grid_random_forest, param_grid, cv=100, scoring="f1", return_train_score=True)
# grid_result.fit(X_train,y_train)
# print(grid_result.best_estimator_)
# print("GridCV Random Forest Train Accuracy: ", grid_result.best_estimator_.score(X_train, y_train))
# print("GridRandom Forest Test Accuracy: ",grid_result.best_estimator_.score(X_test,ytest))
# y_pred = grid_result.best_estimator_.predict(X_test)
# print("Grid random forest Tree F1 score: ",f1_score(ytest, y_pred, average='macro'))
#
#

# SVM tuned
# svm_params = {
#     'kernel': ['linear','rbf','polynomial'],
#     'C' : [0.01,0.01,0.1,0.15,0.2,0.25,0.5,0.75,1,2,10,100],
#     'gamma' : [1,0.1,0.01,0.001,0.0001]
#
# }
# svm_classifier = SVC()
# grid_svm_cv = GridSearchCV(svm_classifier, param_grid=svm_params, cv=5, scoring = 'f1', verbose=1)
# metric = grid_svm_cv.fit(X_train, y_train)
# print(grid_svm_cv.best_estimator_)
# print("Grid SVM Train Accuracy: ", grid_svm_cv.best_estimator_.score(X_train, y_train))
# print("Grid SVM Test Accuracy: ",grid_svm_cv.best_estimator_.score(X_test,ytest))
# y_pred = grid_svm_cv.best_estimator_.predict(X_test)
# print("Grid SVM F1 score: ",f1_score(ytest, y_pred, average='macro'))

