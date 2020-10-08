import utils.utils as utils
import utils.data_utils as du
import numpy as np
import matplotlib.pyplot as plt
import ml_cv.chanvese.multiphase_chanvese as mcv
import time


def test_IOU():
    A = np.zeros((5, 5))
    B = np.zeros((5, 5))

    A[0:3, 1:3] = 1
    B[2:4, 1:4] = 1
    print(A)
    print()
    print(B)
    print(utils.calculate_IOU(A, B))


def test_MIOU():
    print(" Test 3D array")
    A = np.zeros((2, 5, 5))
    B = np.zeros((2, 5, 5))
    A[0, 0:3, 1:3] = 1
    B[0, 2:4, 1:4] = 1
    A[1, 0:3, 1:3] = 1
    B[1, 1:4, 1:4] = 1
    print("A:========")
    print(A)
    print()
    print("B:========")
    print(B)
    MIOU, IOUs = utils.calculate_MIOU(A, B)
    print(MIOU)
    print(IOUs)

    print(" Test 4D array")
    A = np.zeros((3, 2, 5, 5))
    B = np.zeros((3, 2, 5, 5))
    A[0, 0, 0:3, 1:3] = 1
    B[0, 0, 2:4, 1:4] = 1
    A[0, 1, 0:3, 1:3] = 1
    B[0, 1, 1:4, 1:4] = 1

    A[1, 0, 1:3, 1:3] = 1
    B[1, 0, 2:4, 1:4] = 1
    A[1, 1, 0:3, 2:3] = 1
    B[1, 1, 1:4, 1:4] = 1

    A[2, 0, 1:3, 1:3] = 1
    B[2, 0, 2:4, 1:4] = 1
    A[2, 1, 0:3, 0:3] = 1
    B[2, 1, 1:4, 1:4] = 1

    print("A:========")
    print(A)
    print()
    print("B:========")
    print(B)
    MIOU, IOUs = utils.calculate_MIOU(A, B)
    print(MIOU)
    print(IOUs)
    sum = IOUs + IOUs


def test_load_lungs():
    images, targets, targets_sepa, _ = du.load_lungs(img_numbers=1)

    centers = np.array([[50, 100], [160, 75]])
    radiuses = [43, 40]
    phi_arr = mcv._disk(images.shape[1:], 2, centers, radiuses)
    init = np.zeros(phi_arr.shape)
    init[np.where(phi_arr > 0)] = 1
    # util.draw_contours(image, cv[3])

    for i in range(len(images)):
        plt.figure()
        plt.imshow(images[i], cmap="gray")
        centers = np.array([[50, 100], [160, 75]])
        radiuses = [43, 35]

        plt.figure()
        plt.imshow(targets[i])
        plt.show()
        # util.draw_contour(images[i], init)


def test_draw_contour():
    # image, target, target_sepa = du.load_lung()
    image, target = du.load_tumor()
    t1 = time.time()
    utils.draw_contour(image, target)
    # utils.draw_contour(image, target_sepa)
    t2 = time.time()
    # utils.draw_contour1(image, target_sepa)
    t3 = time.time()
    print(t2 - t1)
    print(t3 - t2)


def test_regularize2():
    x = np.linspace(0, 1, 100)
    y = utils.regularize2(0, x)
    plt.figure()
    plt.plot(x, y)
    plt.grid(True)
    plt.xlabel("Probability: x", fontsize=20)
    plt.ylabel("Regularization: f(x)", fontsize=20)
    plt.xticks(size=15)
    plt.yticks(size=15)
    plt.show()


# test_IOU()
# test_MIOU()
test_load_lungs()
# test_draw_contour()
# test_regularize2()
