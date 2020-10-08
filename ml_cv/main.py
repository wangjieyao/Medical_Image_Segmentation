import numpy as np
import matplotlib.pyplot as plt
import time
import ml_cv.chanvese.multiphase_chanvese as mcv
import utils.utils as util
import utils.data_utils as dutils
import ml_cv.mlcv_model as mlcv

def test_ml_chanvese_kidneys():
    n = 20
    radius = [40]
    images, targets, centers = dutils.load_kidneys(n)
    segs = np.empty((n, images.shape[1], images.shape[2]))
    mltimes = 0
    cvtimes = 0
    stime = time.time()
    for i in range(n):
        init_levelset = mcv._disk(images[i].shape, 1, centers[i].reshape(1,2), radius)[0]
        seg, mltime, cvtime = mlcv.ml_chanvese(images[i], targets[i], init_levelset=init_levelset, train_ratio=0.5, max_iter=150, classifer="fknn1", regularize=True)
        segs[i] = seg
        mltimes += mltime
        cvtimes += cvtime
    IOU = util.calculate_IOU(targets, segs)
    etime = time.time()
    print("Total time: {0}, ml time: {1}, cv time: {2}, mean IOU: {3}".format(etime-stime, mltimes, cvtimes, IOU))

def test_ml_chanvese_kidney(i = 2):
    n = 20
    radius = [40]
    images, targets, centers = dutils.load_kidneys(n)
    init_levelset = mcv._disk(images[i].shape, 1, centers[i].reshape(1,2), radius)[0]
    t1 = time. time()
    seg, mltime, cvtime = mlcv.ml_chanvese(images[i], targets[i], init_levelset=init_levelset, train_ratio=0.5, max_iter=160, regularize=True, classifer="fknn")
    t2 = time.time()
    IOU = util.calculate_IOU(targets[i], seg)
    print("i: {0}, IOU: {1}, ml time: {2}, cv time: {3}, total time: {4}".format(i, IOU, mltime, cvtime, t2- t1))
    init = np.zeros(init_levelset.shape)
    init[np.where(init_levelset > 0)] = 1
    util.draw_contour1(images[i], targets[i])
    util.draw_contour_with_target1(images[i], seg, targets[i])

def test_ml_chanvese_lungs():
    n = 20
    radiuses = [50, 50]
    images, targets, targets_sepa, centers = dutils.load_lungs(n)
    mltimes = 0
    cvtimes = 0
    segs = np.empty((n, 2, images.shape[1], images.shape[2]))
    stime = time.time()
    for i in range(n):
        init_levelset = mcv._disk(images[i].shape, 2, centers[i], radiuses)
        seg, mltime, cvtime, _ = mlcv.ml_multiphase_chanvese(images[i], targets[i],targets_sepa[i], init_levelset=init_levelset, train_ratio=0.5, max_iter=1000, regularize=True, classifer="knn")
        segs[i] = seg
        mltimes += mltime
        cvtimes += cvtime
    MIOU, IOUs = util.calculate_MIOU(targets_sepa, segs)
    etime = time.time()
    print("Total time: {0}, ml time: {1}, cv time: {2}, mean IOU: {3}, left lung IOU: {4}, right lung IOU: {5}".format(etime-stime, mltimes, cvtimes, MIOU, IOUs[0], IOUs[1]))

def test_ml_chanvese_lung(i = 6):
    n = 10
    images, targets, targets_sepa, centers = dutils.load_lungs(n, display=True)
    radiuses = [50, 50]
    init_levelset = mcv._disk(images[i].shape, 2, centers[i], radiuses)
    t1 = time.time()
    seg, mltime, cvtime, seg_arr = mlcv.ml_multiphase_chanvese(images[i], targets[i],targets_sepa[i], init_levelset=init_levelset, train_ratio=0.7, max_iter=1000, regularize=True, classifer="fknn", display = True,dt=1.2)
    t2 = time.time()
    MIOU, IOUs = util.calculate_MIOU(targets_sepa[i], seg)
    print("i: {0}, MIOU: {1}, IOUs: {2}, ml time: {3}, cv time: {4}, total time: {5}".format(i, MIOU, IOUs, mltime, cvtime, t2 - t1))
    mious = []
    for seg in seg_arr:
        MIOU, _ = util.calculate_MIOU(targets_sepa[i], seg)
        mious.append(MIOU)
    plt.figure()
    plt.plot(np.arange(0, int(len(seg_arr) * 10), 10), mious)
    plt.grid(True)
    plt.xlabel("Iterations", fontsize=15)
    plt.ylabel("Mean IOU", fontsize=18)
    plt.show()
    init = np.zeros(init_levelset.shape)
    init[np.where(init_levelset > 0)] = 1
    util.draw_contour1(images[i], init)
    util.draw_contour1(images[i], seg)
    # util.draw_contours1(images[i], seg_arr, fn=2, style=1)
    util.draw_contour_with_target1(images[i], seg, targets_sepa[i])

# test_ml_chanvese_kidneys()
# test_ml_chanvese_kidney()
# test_ml_chanvese_lung()
test_ml_chanvese_lungs()
