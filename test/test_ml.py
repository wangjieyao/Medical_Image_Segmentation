import pprint
import operator

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neighbors import KDTree
from sklearn.neighbors.base import KNeighborsMixin, NeighborsBase, SupervisedIntegerMixin
import time
import utils.data_utils as dutil

from ml_cv.ml.fuzzy_knn import FuzzyKNN


def test_fuzzy_knn_iris():
    iris = load_iris()
    # breast = load_breast_cancer()
    dataset = iris

    X = dataset.data
    y = dataset.target
    # t1 = time.time()
    x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=3)
    t1 = time.time()
    fknn = FuzzyKNN(k_neighbors=5)
    fknn.fit(x_train, y_train)
    pred = fknn.predict(x_test)
    print(time.time() - t1)
    print(pred[0])
    print(pred[1])
    accuracy = fknn.score(x_test, y_test)
    print(accuracy)


def test_knn_iris():
    iris = load_iris()
    dataset = iris

    X = dataset.data
    y = dataset.target

    x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=3)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(x_train, y_train)
    pred = knn.predict(x_test)
    print(pred[0])
    print(knn.predict_proba(x_test))
    accuracy = knn.score(x_test, y_test)
    print(accuracy)


def test_fuzzy_knn():
    x = np.linspace(1, 25, 25).reshape((5, 5))
    y = np.zeros((5, 5))
    y[2:4, 3:4] = 1
    # y[0,0] = 2
    y = y.astype(int)

    x = x.flatten()
    y = y.flatten()
    x = x.reshape((-1, 1))

    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=3)

    fknn = FuzzyKNN(k_neighbors=3)
    fknn.fit(x_train, y_train)
    pred = fknn.predict(x_test)
    print(pred[0])
    print(pred[1])
    accuracy = fknn.score(x_test, y_test)
    print(accuracy)


def test_knn():
    x = np.linspace(1, 25, 25).reshape((5, 5))
    y = np.zeros((5, 5))
    y[2:4, 3:4] = 1
    # y[0,0] = 2
    y = y.astype(int)

    x = x.flatten()
    y = y.flatten()
    x = x.reshape((-1, 1))

    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=3)

    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(x_train, y_train)
    pred = knn.predict(x_test)
    print(pred[0])
    print(knn.predict_proba(x_test))
    accuracy = knn.score(x_test, y_test)
    print(accuracy)


test_fuzzy_knn_iris()
# test_knn_iris()
# test_fuzzy_knn()
# test_knn()
