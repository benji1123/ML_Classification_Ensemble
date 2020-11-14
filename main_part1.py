#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 10:24:35 2020

@author: benjaminli
"""
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import plot_precision_recall_curve
import matplotlib.pyplot as plt

import helper_part1 as helper

SEED = 878  #0878
LR = 0.1
EPOCHS = 1000
NUM_NEIGHBOURS = 2

## WHAT TO RUN
run = {
       "logreg man": False,
       "logreg sk": False,
       "knear man": True,
       "knear sk": False,
       }


# load data
bc_data = load_breast_cancer(as_frame=True)  # panda dataframe
X_train, X_test, y_train, y_test = helper.split_data(bc_data, SEED)
sc = StandardScaler()  # feature normalization
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
theta = np.random.rand(30)

# Logistic Regression manual implementation
if run["logreg man"]:
    for i in range(EPOCHS):
        h = helper.sigmoid(X_train, theta.T)
        grad = helper.get_grad(X_train, y_train, h)
        theta = helper.update_theta(theta, LR, grad)
    test_error = helper.get_cost(X_test, y_test, theta)
    training_error = helper.get_cost(X_train, y_train, theta)
    print(f"\nLOGISTIC REGRESSION\n-------------------"
          f"\nmanual test error: {test_error}"
          f"\nmanual training error: {training_error}")
    # PR Curve
    predictions = helper.get_prob(X_test, theta)
    precision, recall, thresholds = precision_recall_curve(y_test, predictions)
    plt.plot(recall, precision)
    plt.title("Manual LogReg PR Curve")
    
    
# Logistic Regression sk-learn implementation
if run["logreg sk"]:
    LR_scipy_model= LogisticRegression()
    LR_scipy_model.fit(X_train, y_train)
    test_error = 1.0 - LR_scipy_model.score(X_test, y_test)
    training_error = 1.0 - LR_scipy_model.score(X_train, y_train)
    print(f"\nsk test error: {test_error}"
          f"\nsk training error: {training_error}")
    # PR Curve
    disp = plot_precision_recall_curve(LR_scipy_model, X_test, y_test)
    disp.ax_.set_title('SK LogReg PR Curve')

# K-Nearest manual implementation
if run["knear man"]:
    test_predictions = helper.get_predictions_knn(X_train, y_train, X_test, NUM_NEIGHBOURS)
    train_predictions = helper.get_predictions_knn(X_train, y_train, X_train, NUM_NEIGHBOURS)
    test_error = helper.get_knn_error(test_predictions, y_test)
    training_error = helper.get_knn_error(train_predictions, y_train)
    print(f"\nK-NEAREST NEIGHBOURS\n--------------------"
          f"\nmanual test error: {test_error}"
          f"\nmanual training error: {training_error}")
    # PR Curve
    precision, recall, thresholds = precision_recall_curve(test_predictions, y_test)
    plt.plot(recall, precision)
    plt.title("Manual KNN PR Curve")
# K-Nearest sk-learn implementation
if run["knear sk"]:
    KNN_scipy_model = KNeighborsClassifier(n_neighbors=NUM_NEIGHBOURS)
    KNN_scipy_model.fit(X_train, y_train)
    test_error = 1 - KNN_scipy_model.score(X_test, y_test)
    training_error = 1 - KNN_scipy_model.score(X_train, y_train)
    print(f"\nsk test error: {test_error}"
          f"\nsk training error: {training_error}")
    disp = plot_precision_recall_curve(KNN_scipy_model, X_test, y_test)
    disp.ax_.set_title('SK K-Nearest PR Curve')
