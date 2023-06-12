# Mental-Health Prediction
Project of CS 725 course at IIT Bombay, Autumn 2022.

This project is completed as a part of the course Foundations of Machine Learning (CS 725) at Indian Institute of Technology Bombay in Autumn, 2022. The course is offered by the Dept. of Computer Science and Engineering of the institute. The project objective has been accomplished by a team of five M.Tech first year students.



# Problem Statement
The objective of this project is to determine the mental state of an individual on the basis of its mental and physical wellbeing and its environment, and indicate whether professional help is required or not. For this project, we used various machine learning algorithms including Decision trees, Stacking, Logistic Regression, SVM, Random Forest, Neural Networks, Gradient Boost and Adaboost. The resulting outcomes are compared and used for selecting the appropriate model for the
predictions.


# Architecture
![image](https://github.com/iamtatha/Mental-Health/assets/57251093/65f66392-0dac-4c4d-8b45-d087e8473cbe)

A high level methodology is shown in figure. The input dataset is preprocessed. For preprocessing, data cleaning, data encoding and feature selection are used. After the data is preprocessed, it is used to train various models which we have used, including logistic regression, decision trees, random forests, stacking, SVM, Neural network and more. Based on the accuracy, a model is selected and predictions are made using the selected model.


# Dataset
Dataset of OSMI/OSMI Mental Health in Tech Survey (https://osmihelp.org/research) is used for the objective.
Dataset is publicly available and it has the survey information from 2014 to 2021. The dataset has a wide variety of questions asked in relation to mental health.

![image](https://github.com/iamtatha/Mental-Health/assets/57251093/81999e24-1d17-4919-bf67-3935cc055e2d)


# Results
![image](https://github.com/iamtatha/Mental-Health/assets/57251093/ce442a10-dd62-4b93-92c3-e07cb64da72c)
The results of all the models ares shown in table 6 with Accuracy on test set, precision, recall and f1 scores. The models are executed on multiple iterations to achieve the best possible results. The most sound results were found from Support Vector Machines with an accuracy of 89% whereas other models aren’t far behind.


All the codes, data and report are shared in the repository.

