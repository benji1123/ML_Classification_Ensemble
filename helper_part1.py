#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 11:41:07 2020

@author: benjaminli
"""
import numpy as np
from numpy import ndarray
from sklearn.model_selection import train_test_split

CATEGORY_INDEX = 2


### Logistic Regression Helper 


def update_theta(theta, lr, grad):
    """weight-update step."""
    return theta - lr * grad


def split_data(data, seed=0):
    X_train, X_test, y_train, y_test = train_test_split(
    data['data'], data['target'], test_size=0.3, random_state=seed)
    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy()
    return X_train, X_test, y_train, y_test


def get_cost(X, y, theta):
    """cross-entropy (log-loss) cost function."""
    prob = get_prob(X, theta)
    m = X.shape[0]
    cost = np.sum(y * np.log(prob) + (1 - y) * np.log(1 - prob)) * (-1/m)
    return cost


def get_grad(X, y, h):
    grad = np.dot(X.T, (h-y).T) / y.shape[0]
    return grad


def get_prob(X, theta):
    """probability that X is in some class."""
    return sigmoid(X, theta)


def sigmoid(X, theta):
    """sigmoid activation function."""
    z = np.matmul(X, theta)
    return 1 / (1 + np.exp(-z))


### K-Nearest Helper functions


def get_euclidean_dist(p1, p2):
    p1 = np.array(p1)
    p2 = np.array(p2)
    return np.linalg.norm(p1 - p2)


def get_k_nearest(X, y, source, k, get_dist=get_euclidean_dist):
    """return k-nearest neighbours in some dataset"""
    neighbours = []
    for i, point in enumerate(X):
        dist = get_dist(source, point)
        neighbours.append((point, dist, y[i]))
    neighbours.sort(key=lambda n: n[1])  # sort by distance (descending order)
    m = len(X)
    if len(neighbours) < k:
        ValueError(f"cannot return {k} neighbours in dataset with {m} points")
    return neighbours[:k]


def get_category_score_knn(neighbours):
    score = 0.0
    for n in neighbours:
        score += n[CATEGORY_INDEX]
    score = score / len(neighbours)
    if score >= 0.5:
        return 1, score
    else:
        return 0, 1.0 - score


def get_predictions_knn(X, y, test_points, k=5):
    predictions = []
    for point in test_points:
        neighbours = get_k_nearest(X, y, point, k)
        category, score = get_category_score_knn(neighbours)
        predictions.append((category))
    return predictions


def get_knn_error(predictions, labels):
    successes = 0
    for prediction, label in zip(predictions, labels):
        if prediction == label:
            successes += 1
    return 1 - successes / len(labels)

        
    