import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from itertools import chain
import ml_cv.chanvese.multiphase_chanvese as mcv
import utils.utils as util
from ml_cv.ml.fuzzy_knn import FuzzyKNN
import ml_cv.chanvese.chanvese as cv


def ml_multiphase_chanvese(image, target, target_sepa, train_ratio=0.5, k_neighbor=9, regularize=True, nu=0.7,
                           init_levelset=None, max_iter=1000, classifer="fknn", display=True, dt=1):
    """
    machine learning + multiphase chanvese 模型 （针对多分类，使用2个水平集函数，最多分类4类）
    通过classifier选择算法对像素进行分类, 使用multiphase chanvese对分类结果进行分割；
    :param image:         原始图像
    :param target:        ground truth
    :param target_sepa:
    :param train_ratio:   训练集占比
    :param k_neighbor:
    :param regularize:    是否使用正则化函数
    :param nu: multiphase chanvese 参数
    :param init_levelset: 初始化水平集
    :param max_iter:      最大的迭代次数
    :param classifer:     "knn": KNN; "fknn": Fuzzy KNN; 其他：不采用machine learning算法，相当于原始的multiphase chanvese
    :param display:       是否展示结果图
    :param dt:            multiphase chanvese 参数
    :return:
    """
    t1 = time.time()
    x = np.array(list(chain(*image)))
    y = np.array(list(chain(*target))).astype(int)
    x = x.reshape((len(x), 1))
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=(1 - train_ratio), random_state=3)
    if classifer == "fknn":
        fknn = FuzzyKNN(k_neighbors=k_neighbor, leaf_size=1, n_jobs=1)
        fknn.fit(x_train, y_train)
        pred, weihts = fknn.predict(x)
        scores = pred
        if regularize:
            probabilities = np.amax(weihts, axis=1)
            scores = util.regularize1(pred, probabilities)
        scores = scores.reshape(image.shape)
    elif classifer == "knn":
        knn = KNeighborsClassifier(n_neighbors=k_neighbor, leaf_size=1, n_jobs=1)
        knn.fit(x_train, y_train)
        pred = knn.predict(x)
        weights = knn.predict_proba(x)
        scores = pred
        if regularize:
            probabilities = np.amax(weights, axis=1)
            scores = util.regularize1(pred, probabilities)
        scores = scores.reshape(image.shape)
    else:
        scores = image
    t2 = time.time()
    if display:
        plt.figure()
        plt.imshow(scores)
        plt.axis('off')
        plt.show()
        segmentation, _, seg_arr = mcv.multiphase_chanvese_display(scores, nu=nu, max_iter=max_iter, dt=dt,
                                                                   init_levelset=init_levelset)
    else:
        segmentation, _, seg_arr = mcv.multiphase_chanvese(scores, nu=nu, max_iter=max_iter, dt=dt,
                                                           init_levelset=init_levelset)
    t3 = time.time()
    return segmentation, t2 - t1, t3 - t2, seg_arr


def ml_chanvese(image, target, train_ratio=0.5, k_neighbor=9, regularize=True, mu=0.6, init_levelset=None, max_iter=200,
                classifer="fknn"):
    """
    machine learning + chanvese 模型 （二分类）
    通过classifier选择算法对像素进行二分类, 使用chanvese对分类结果进行分割
    :param image:         原始图像
    :param target:        ground truth
    :param train_ratio:   训练集占比
    :param k_neighbor:
    :param regularize:    是否对分类结果使用正则化函数
    :param mu:            chanvese模型参数
    :param init_levelset: 初始化水平集
    :param max_iter:      最大迭代数
    :param classifer:     "knn": KNN; "fknn": Fuzzy KNN; 其他：不采用machine learning算法，相当于原始的chanvese
    :return:
    """
    start = time.time()
    # 1. train the classifier and get the scores
    x = np.array(list(chain(*image)))
    y = np.array(list(chain(*target))).astype(int)
    x = x.reshape((-1, 1))
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=(1 - train_ratio), random_state=3)
    if classifer == "fknn":
        fknn = FuzzyKNN(k_neighbors=k_neighbor, leaf_size=1, n_jobs=4)
        fknn.fit(x_train, y_train)
        pred, weihts = fknn.predict(x)
        scores = pred
        if regularize:
            probabilities = np.amax(weihts, axis=1)
            scores = util.regularize1(pred, probabilities)
        scores = scores.reshape(image.shape)
    elif classifer == "knn":
        knn = KNeighborsClassifier(n_neighbors=k_neighbor, leaf_size=1, n_jobs=4)
        knn.fit(x_train, y_train)
        pred = knn.predict(x)
        weights = knn.predict_proba(x)
        scores = pred
        if regularize:
            probabilities = np.amax(weights, axis=1)
            scores = util.regularize1(pred, probabilities)
        scores = scores.reshape(image.shape)
    else:
        scores = image
    t1 = time.time()
    segmentation, _, seg_arr = cv.chan_vese(scores, mu=mu, init_levelset=init_levelset, max_iter=max_iter, dt=0.2)
    t2 = time.time()
    return segmentation, t1 - start, t2 - t1


def ml_multiphase_chanvese_one_classifier(images, targets, target_sepas, train_ratio=0.5, k_neighbor=9, regularize=True,
                                          nu=0.7, init_levelsets=None, max_iter=1000, classifer="fknn", display=True,
                                          dt=1):
    """
    使用 machine learning + multiphase chanvese 模型，对于多张图片，仅仅训练一个分类器
    多张图片作为输入，从输入中随机分割出一部分像素作为训练集，训练后的分类器对所有像素进行分类，再使用multiphase chanvese
    对分类结果进行分割
    :param images:        原始的多张图片
    :param targets:       ground truth
    :param target_sepas:
    :param train_ratio:   训练集占比
    :param k_neighbor:
    :param regularize:    是否使用正则化函数
    :param nu: multiphase chanvese 参数
    :param init_levelsets: 初始化水平集
    :param max_iter:      最大的迭代次数
    :param classifer:     "knn": KNN; "fknn": Fuzzy KNN; 其他：不采用machine learning算法，相当于原始的multiphase chanvese
    :param display:       是否展示结果图
    :param dt:            multiphase chanvese 参数
    :return:
    """
    t1 = time.time()
    x = images.flatten()
    y = targets.flatten().astype(int)
    x = x.reshape((len(x), 1))
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=(1 - train_ratio), random_state=3)
    if classifer == "fknn":
        fknn = FuzzyKNN(k_neighbors=k_neighbor, leaf_size=1, n_jobs=4)
        fknn.fit(x_train, y_train)
        pred, weihts = fknn.predict(x)
        scores = pred
        if regularize:
            probabilities = np.amax(weihts, axis=1)
            scores = util.regularize1(pred, probabilities)
        scores = scores.reshape(images.shape)
    elif classifer == "knn":
        knn = KNeighborsClassifier(n_neighbors=k_neighbor, leaf_size=1, n_jobs=4)
        knn.fit(x_train, y_train)
        pred = knn.predict(x)
        weights = knn.predict_proba(x)
        scores = pred
        if regularize:
            probabilities = np.amax(weights, axis=1)
            scores = util.regularize1(pred, probabilities)
        scores = scores.reshape(images.shape)
    else:
        scores = images
    segmentations = np.empty(target_sepas.shape)
    t2 = time.time()
    for i in range(images.shape[0]):
        if display:
            segmentation, _, seg_arr = mcv.multiphase_chanvese_display(scores[i], nu=nu, max_iter=max_iter, dt=dt,
                                                                       init_levelset=init_levelsets[i])
        else:
            segmentation, _, seg_arr = mcv.multiphase_chanvese(scores, nu=nu, max_iter=max_iter, dt=dt,
                                                               init_levelset=init_levelsets[i])
        segmentations[i] = segmentation
    t3 = time.time()
    return segmentations, t2 - t1, t3 - t2


def ml_chanvese_one_classifier(images, targets, train_ratio=0.5, k_neighbor=9, regularize=True, mu=0.6,
                               init_levelsets=None, max_iter=200, classifer="fknn"):
    """
    使用 machine learning + chanvese 模型，对于多张图片，仅仅训练一个分类器
    多张图片作为输入，从输入中随机分割出一部分像素作为训练集，训练后的分类器对所有像素进行二分类，再使用chanvese对分类结果进行分割
    :param images:        原始多张图片
    :param targets:       ground truth
    :param train_ratio:   训练集占比
    :param k_neighbor:
    :param regularize:    是否对分类结果使用正则化函数
    :param mu:            chanvese模型参数
    :param init_levelsets: 初始化水平集
    :param max_iter:      最大迭代数
    :param classifer:     "knn": KNN; "fknn": Fuzzy KNN; 其他：不采用machine learning算法，相当于原始的chanvese
    :return:
    """
    t1 = time.time()
    x = images.flatten()
    y = targets.flatten().astype(int)
    x = x.reshape((len(x), 1))
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=(1 - train_ratio), random_state=3)
    if classifer == "fknn":
        fknn = FuzzyKNN(k_neighbors=k_neighbor, leaf_size=1, n_jobs=4)
        fknn.fit(x_train, y_train)
        pred, weihts = fknn.predict(x)
        scores = pred
        if regularize:
            probabilities = np.amax(weihts, axis=1)
            scores = util.regularize1(pred, probabilities)
        scores = scores.reshape(images.shape)
    elif classifer == "knn":
        knn = KNeighborsClassifier(n_neighbors=k_neighbor, leaf_size=1, n_jobs=4)
        knn.fit(x_train, y_train)
        pred = knn.predict(x)
        weights = knn.predict_proba(x)
        scores = pred
        if regularize:
            probabilities = np.amax(weights, axis=1)
            scores = util.regularize1(pred, probabilities)
        scores = scores.reshape(images.shape)
    else:
        scores = images
    segmentations = np.empty(targets.shape)
    t2 = time.time()
    for i in range(images.shape[0]):
        segmentation, _, seg_arr = cv.chan_vese(scores[i], mu=mu, init_levelset=init_levelsets[i], max_iter=max_iter,
                                                dt=0.2)
        segmentations[i] = segmentation
    t3 = time.time()
    return segmentations, t2 - t1, t3 - t2
