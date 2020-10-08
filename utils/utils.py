import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage import measure, draw
import math

def calculate_IOU(target, prediction):
    intersection = np.logical_and(target, prediction)
    union = np.logical_or(target, prediction)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score


def calculate_MIOU(targets, predictions):
    """
    calculate mean IOU
    :param target:   np.array(numbers, classes, image height, image width) or  np.array(classes, image height, image width)
    :param prediction:
    :return:
    """
    if np.ndim(targets) != np.ndim(predictions):
        raise Exception("Targets and Predictions should be the same size.")
    if np.ndim(targets) != 3 and np.ndim(targets) != 4 :
        raise Exception("Targets or Predictions should be a 3D or 4D array.")
    if np.ndim(targets) == 3:
        IOUs = []
        for i in range(targets.shape[0]):
            target = targets[i]
            prediction = predictions[i]
            iou = calculate_IOU(target, prediction)
            IOUs.append(iou)
        MIOU = np.nanmean(IOUs)
        return MIOU, IOUs

    n, classes = targets.shape[0:2]
    IOUs = []
    intersection = np.logical_and(targets, predictions)
    union = np.logical_or(targets, predictions)
    for i in range(classes):
        in_count = np.sum(intersection[:,i,:,:])
        un_count = np.sum(union[:,i,:,:])
        IOUs.append(in_count / un_count)
    MIOU = np.nanmean(IOUs)

    return MIOU, np.array(IOUs)

def regularize1(classes, probalities):
    # scores = np.amax(weihts, axis=1)
    return  classes + (1 + (2 * (1 - np.array(probalities)) - 1) ** 3) / 2.


def regularize2(classes, probalities):
    # scores = np.amax(weihts, axis=1)
    return  classes + 0.5 *(1 - np.cos(np.pi * (1-probalities)) ** 5)

def regularize3(classes, probalities):
    # scores = np.amax(weihts, axis=1)
    return  classes + 1 - probalities


def draw_contour(image, seg_image):
    binary = np.uint8(seg_image)
    if np.ndim(seg_image) == 2 :
        _ ,contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = np.vstack(contours).squeeze()
        plt.figure()
        plt.imshow(image, plt.cm.gray)
        plt.plot(contours[:,0],contours[:,1], c='r',lw=3)
        plt.figure()
        plt.imshow(binary)
        plt.show()
        return

    n = len(seg_image)
    _, ax = plt.subplots(nrows=1, ncols=n, figsize=(1 * 60, 40))
    _, ax1 = plt.subplots(nrows=1, ncols=n, figsize=(1 * 60, 40))
    colors = ['r','m','b']
    for i in range(n):
        _ ,contours, _ = cv2.findContours(binary[i], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = np.vstack(contours).squeeze()
        ax[i].imshow(image, plt.cm.gray)
        ax[i].plot(contours[:,0],contours[:,1], c=colors[i],lw=10)
        ax1[i].imshow(seg_image[i])
    plt.show()


def draw_contours(image, seg_image_arr, fn=2):
    binary_arr = np.uint8(seg_image_arr)
    if np.ndim(seg_image_arr) == 3:
        for i in range(len(binary_arr)):
            _ ,contours, _ = cv2.findContours(binary_arr[i], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = np.vstack(contours).squeeze()
            plt.figure()
            plt.imshow(image, plt.cm.gray)
            plt.plot(contours[:,0],contours[:,1], c='r',lw=3)
            # plt.figure()
            # plt.imshow(binary_arr[i])
            plt.show()
        return
    # fn = seg_image_arr.shape[1]
    _, ax = plt.subplots(nrows=1, ncols=fn, figsize=(1 * 60, 40))
    colors = ['r','m','b']
    for i in range(len(binary_arr)):
        _, ax = plt.subplots(nrows=1, ncols=fn, figsize=(1 * 60, 40))
        _, ax1 = plt.subplots(nrows=1, ncols=fn, figsize=(1 * 60, 40))
        for j in range(fn):
            _ ,contours, _ = cv2.findContours(binary_arr[i,j], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = np.vstack(contours).squeeze()
            ax[j].imshow(image, plt.cm.gray)
            ax[j].scatter(contours[:,0],contours[:,1], c=colors[j],lw=10)
            ax1[j].imshow(binary_arr[i,j])
        plt.show()

def draw_contour1(image, seg_image):
    binary = np.uint8(seg_image)
    if np.ndim(seg_image) == 2 :
        plt.figure()
        contours = measure.find_contours(binary, 0.2)
        # contours = np.vstack(contours).squeeze()
        plt.imshow(image, plt.cm.gray)
        for _, contour in enumerate(contours):
            plt.plot(contour[:, 1], contour[:, 0], linewidth=5, c='red')
        plt.xticks([])
        plt.yticks([])
        plt.show()
        return
    n = len(seg_image)
    _, ax = plt.subplots(nrows=1, ncols=n, figsize=(1 * 60, 40))
    # _, ax1 = plt.subplots(nrows=1, ncols=n, figsize=(1 * 60, 40))
    colors = ['r','y','b']
    for i in range(n):
        contours = measure.find_contours(binary[i], 0.2)
        # _ ,contours, _ = cv2.findContours(binary[i], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # contours = np.vstack(contours).squeeze()
        ax[i].imshow(image, plt.cm.gray)
        for _, contour in enumerate(contours):
            ax[i].plot(contour[:, 1], contour[:, 0], linewidth=5, c=colors[i])
    plt.axis('off')
        # ax1[i].imshow(seg_image[i])
    plt.show()

def draw_contour_with_target1(image, seg_image, target):
    seg_image = np.uint8(seg_image)
    target = np.uint8(target)
    if np.ndim(seg_image) == 2 :
        plt.figure()
        contours = measure.find_contours(seg_image, 0.2)
        t_con = measure.find_contours(target, 0.2)
        plt.imshow(image, plt.cm.gray)
        for _, contour in enumerate(contours):
            plt.plot(contour[:, 1], contour[:, 0], linewidth=3, c='red')
        for _, tc in enumerate(t_con):
            plt.plot(tc[:, 1], tc[:, 0], linewidth=3, c='b', linestyle='--')
        plt.axis('off')
        plt.show()
        return
    n = len(seg_image)
    plt.figure()
    # _, ax = plt.subplots(nrows=1, ncols=n, figsize=(1 * 60, 40))
    # _, ax1 = plt.subplots(nrows=1, ncols=n, figsize=(1 * 60, 40))
    colors = ['r','y','g','b']
    plt.imshow(image, plt.cm.gray)
    for i in range(n):
        contours = measure.find_contours(seg_image[i], 0.2)
        for _, contour in enumerate(contours):
            plt.plot(contour[:, 1], contour[:, 0], linewidth=3, c=colors[i])
        t_con = measure.find_contours(target[i], 0.2)
        for _, contour in enumerate(t_con):
            plt.plot(contour[:, 1], contour[:, 0], linewidth=3, c=colors[i+2], linestyle='--')
        # ax1[i].imshow(seg_image[i])
    plt.axis('off')
    plt.show()

def draw_contours1(image, seg_image_arr, fn=2, style=1):
    """

    :param image:
    :param seg_image_arr:
    :param fn:
    :param style:  1：show  one image  2: show multiple images
    :return:
    """
    binary_arr = np.uint8(seg_image_arr)
    if np.ndim(seg_image_arr) == 3:
        if style != 1:
            for i in range(len(binary_arr)):
                contours = measure.find_contours(binary_arr[i], 0.2)
                plt.imshow(image, plt.cm.gray)
                for _, contour in enumerate(contours):
                    plt.plot(contour[:, 1], contour[:, 0], linewidth=3, c='red')
                plt.show()
            return
        else:
            ncols = 4
            nrows = math.ceil(len(seg_image_arr) / ncols)
            _, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(1 * 60, 40))
            for i in range(binary_arr.shape[0]):
                contours = measure.find_contours(binary_arr[i], 0.2)
                ax[int(i // ncols)][int(i % ncols)].imshow(image, plt.cm.gray)
                ax[int(i // ncols)][int(i % ncols)].set_title("Iter: {0}".format(i * 10), fontsize=100)
                for _, contour in enumerate(contours):
                    ax[int(i // ncols)][int(i % ncols)].plot(contour[:, 1], contour[:, 0], linewidth=3, c='red')
            ax[0][0].set_title("Initialization", fontsize=100)
            plt.axis('off')
            plt.show()
            return

    # fn = seg_image_arr.shape[1]

    colors = ['r','y','b']
    if style != 1:
        _, ax = plt.subplots(nrows=1, ncols=fn, figsize=(1 * 60, 40))
        for i in range(len(binary_arr)):
            _, ax = plt.subplots(nrows=1, ncols=fn, figsize=(1 * 60, 40))
            _, ax1 = plt.subplots(nrows=1, ncols=fn, figsize=(1 * 60, 40))
            for j in range(fn):
                _ ,contours, _ = cv2.findContours(binary_arr[i,j], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                contours = np.vstack(contours).squeeze()
                ax[j].imshow(image, plt.cm.gray)
                ax[j].scatter(contours[:,0],contours[:,1], c=colors[j],lw=10)
                ax1[j].imshow(binary_arr[i,j])
            plt.show()
    else:
        ncols = 4
        nrows = math.ceil(len(seg_image_arr) / ncols)
        _, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(1 * 60, 60))
        for i in range(len(binary_arr)):
            for j in range(fn):
                contours = measure.find_contours(binary_arr[i,j], 0.2)
                ax[int(i // ncols)][int(i % ncols)].imshow(image, plt.cm.gray)
                ax[int(i // ncols)][int(i % ncols)].set_title("Iter: {0}".format(i * 100), fontsize=100)
                for _, contour in enumerate(contours):
                    ax[int(i // ncols)][int(i % ncols)].plot(contour[:, 1], contour[:, 0], linewidth=5, c=colors[j])
        ax[0][0].set_title("Initialization", fontsize=100)
        plt.axis('off')
        plt.show()



