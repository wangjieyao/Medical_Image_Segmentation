import numpy as np
from scipy.ndimage import distance_transform_edt as distance


def _chessboard(image_size, chessboard_ind, square_size, fn):
    """
    Generate the checkerboard shape level set function: height: init_shape[0]+1, width: init_shape[1]+1
    生成棋盘形状的水平集
    e.g. fn=2 square_size=3 image_size
    result[0]=[[ 0.    0.    0.    0.   -1.   -1.   -1.  ]
         [ 0.    0.75  0.75  0.   -1.   -1.   -1.  ]
         [ 0.    0.75  0.75  0.   -1.   -1.   -1.  ]
         [ 0.    0.    0.    0.    0.    0.    0.  ]
         [-1.   -1.   -1.    0.    0.75  0.75  0.  ]
         [-1.   -1.   -1.    0.    0.75  0.75  0.  ]
         [-1.   -1.   -1.    0.    0.    0.    0.  ]]
    :param image_size:     图片尺寸  [row, col]
    :param chessboard_ind: 从指定位置才开始绘制棋盘形状  e.g. np.array([[[20,row-20],[20, col_cen]],[[20,row-20],[col_cen + 40, col-20]]])
    :param square_size:    棋盘格子尺寸 e.g. 10
    :param fn:             水平集函数数量 e.g. 2
    :return:  shape: [fn, row, col]
    """
    yv = np.arange(image_size[0]).reshape(image_size[0], 1)
    xv = np.arange(image_size[1])
    board = (np.sin(np.pi / square_size * yv) *
             np.sin(np.pi / square_size * xv))
    result = np.zeros((fn, image_size[0], image_size[1]))
    for i in range(fn):
        r_s, r_e = chessboard_ind[i, 0, 0], chessboard_ind[i, 0, 1]
        c_s, c_e = chessboard_ind[i, 1, 0], chessboard_ind[i, 1, 1]
        result[i, r_s:r_e, c_s:c_e] = board[r_s:r_e, c_s:c_e]

    return result

def _disk(image_size, fn, center, radius):
    """
    Generates a disk level set function.
    生成圆形的水平集函数
    :param image_size:
    :param fn:
    :param center:
    :param radius:
    :return:
    """
    res = np.ones((fn, image_size[0], image_size[1]))
    for i in range(fn):
        centerY = int(center[i,1])
        centerX = int(center[i,0])
        res[i, centerY, centerX] = 0.
        res[i] = (radius[i] - distance(res[i])) / radius[i]
    return res

def _large_disk(image_size, fn):
    """
    Generates a disk level set function.The disk covers the whole image along its smallest dimension.
    生成圆形的水平集函数
    :param image_size:  [row, col]
    :param fn: 水平集函数数量
    :return:   [fn, row, col]
    """
    if fn < 2:
        res = np.ones(image_size)
        centerY = int((image_size[0] - 1) / 2)
        centerX = int((image_size[1] - 1) / 2)
        res[centerY, centerX] = 0.
        radius = float(min(0.5 * centerX, 0.5 * centerY))
        return [(radius - distance(res)) / radius]

    res = np.ones((2, image_size[0], image_size[1]))
    radius = float(min(0.25 * image_size[0], 0.25 * image_size[1]))
    for i in range(2):
        centerY = int((image_size[0] - 1) / 4)
        centerX = int((image_size[1] - 1) / 4 + i * 2 * radius)
        res[i, centerY, centerX] = 0.
        res[i] = (radius - distance(res[i])) / radius
    return res


def _heavyside(x, eps=1.):
    """
    Returns the result of a regularised heavyside function of the input value.
    返回Heavyside函数（单位越阶函数）结果，它会逼近于 0.5 + 1/π * arctan(kx)
    :param x:
    :param eps:
    :return:
    """
    return 1 * (x > 0)


def _delta(x, eps=1.):
    """
    Returns the result of a regularised dirac function of the input value.
    返回狄拉克δ函数的结果
    :param x:
    :param eps:
    :return:
    """
    return eps / np.pi / (eps ** 2 + x ** 2)


def _curvature_energy(phi_arr):
    """
    Calculate the energy of the curvature term
    计算曲率项的能量
    :param phi_arr:
    :return:
    """
    K = np.zeros(phi_arr.shape)
    for i in range(phi_arr.shape[0]):
        P = np.pad(phi_arr[i], 1, mode='edge')
        fy = (P[2:, 1:-1] - P[:-2, 1:-1]) / 2.0
        fx = (P[1:-1, 2:] - P[1:-1, :-2]) / 2.0
        fyy = P[2:, 1:-1] + P[:-2, 1:-1] - 2 * phi_arr[i]
        fxx = P[1:-1, 2:] + P[1:-1, :-2] - 2 * phi_arr[i]
        fxy = .25 * (P[2:, 2:] + P[:-2, :-2] - P[:-2, 2:] - P[2:, :-2])
        grad2 = fx ** 2 + fy ** 2
        k = ((fxx * fy ** 2 - 2 * fxy * fx * fy + fyy * fx ** 2) /
             (grad2 * np.sqrt(grad2) + 1e-8))
        K[i] = k
    return K


def _curvatures(phi):
    """
    Calculate the curvatures of the level set functions
    :param phi:
    :return:
    """
    eta = 1e-16
    K = np.zeros(phi.shape)
    C_sum = np.zeros(phi.shape)
    for i in range(phi.shape[0]):
        P = np.pad(phi[i], 1, mode='edge')

        phixp = P[1:-1, 2:] - P[1:-1, 1:-1]
        phixn = P[1:-1, 1:-1] - P[1:-1, :-2]
        phix0 = (P[1:-1, 2:] - P[1:-1, :-2]) / 2.0

        phiyp = P[2:, 1:-1] - P[1:-1, 1:-1]
        phiyn = P[1:-1, 1:-1] - P[:-2, 1:-1]
        phiy0 = (P[2:, 1:-1] - P[:-2, 1:-1]) / 2.0

        C1 = 1. / np.sqrt(eta + phixp ** 2 + phiy0 ** 2)
        C2 = 1. / np.sqrt(eta + phixn ** 2 + phiy0 ** 2)
        C3 = 1. / np.sqrt(eta + phix0 ** 2 + phiyp ** 2)  # AI J
        C4 = 1. / np.sqrt(eta + phix0 ** 2 + phiyn ** 2)

        K[i] = (P[1:-1, 2:] * C1 + P[1:-1, :-2] * C2 +
                P[2:, 1:-1] * C3 + P[:-2, 1:-1] * C4)
        C_sum[i] = (C1 + C2 + C3 + C4)

    return K, C_sum


def _calculate_averages(image, phi_arr, fn):
    """
    Calculate the average pixels 'inside' and 'outside'.
    计算分割内外的平均像素
    :param image:
    :param phi_arr:
    :param fn:
    :return:
    """
    Hphi = _heavyside(phi_arr)
    Hinv = 1 - Hphi

    def average(h):
        result = np.sum(image * h)
        sum = np.sum(h)
        if sum != 0:
            result /= sum
        return result

    if fn == 1:
        C1 = average(Hphi[0])
        C0 = average(Hinv[0])
        return (C1, None, None, C0), Hphi, Hinv

    C11 = average(Hphi[0] * Hphi[1])
    C01 = average(Hinv[0] * Hphi[1])
    C10 = average(Hphi[0] * Hinv[1])
    C00 = average(Hinv[0] * Hinv[1])
    return (C11, C01, C10, C00), Hphi, Hinv


def _calculate_phi(image, phi_arr, nu, fn, dt):
    """
    Returns the variation of level set 'phi' based on algorithm parameters.
    计算水平集中的phi变量
    :param image:
    :param phi_arr:
    :param nu:
    :param fn:
    :param dt:
    :return:
    """
    def cal_new_phi(phi, average_term, k, c_sum):
        new_phi = (phi + dt * _delta(phi) * (nu * k + average_term))
        return new_phi / (1 + nu * dt * _delta(phi) * c_sum)

    if fn == 1:
        K, C_sum = _curvatures(phi_arr)
        (C11, C01, C10, C00), Hphi, Hinv = _calculate_averages(image, phi_arr, fn)
        average_term = (- (image - C11) ** 2 + (image - C00) ** 2)
        return cal_new_phi(phi_arr[0], average_term, K[0], C_sum[0])

    temp_phi = phi_arr
    new_phi = np.empty(phi_arr.shape)
    for i in range(2):
        K, C_sum = _curvatures(phi_arr)
        (C11, C01, C10, C00), Hphi, Hinv = _calculate_averages(image, temp_phi, fn)

        """
        [1] = (C01 - C11)(C01 +C11 - 2U)*H(φ2) + (C10 - C00)(C10 +C00 - 2U)*（1-H(φ2)）
        [2] = (C10 - C11)(C10 +C11 - 2U)*H(φ1) + (C01 - C00)(C01 +C00 - 2U)*（1-H(φ1)）
        """
        if i == 0:
            average_term = (C01 - C11) * (C01 + C11 - 2 * image) * Hphi[1]
            average_term += (C00 - C10) * (C10 + C00 - 2 * image) * Hinv[1]
        else:
            average_term = (C10 - C11) * (C10 + C11 - 2 * image) * Hphi[0]
            average_term += (C00 - C01) * (C01 + C00 - 2 * image) * Hinv[0]
        new_phi[i] = cal_new_phi(temp_phi[i], average_term, K[i], C_sum[i])

    return new_phi


def _energy(image, phi_arr, nu, fn):
    """
    Return the total energy
    返回总能量
    :param image:
    :param phi_arr:
    :param nu:
    :param fn:
    :return:
    """
    (C11, C01, C10, C00), Hphi, Hinv = _calculate_averages(image, phi_arr, fn)
    if fn < 2:
        avgenergy = (image - C11) ** 2 * Hphi[0] + (image - C00) ** 2 * Hinv[0]
        lenenergy = _curvature_energy(phi_arr)
        return np.sum(avgenergy) + np.sum(nu * lenenergy)

    avgenergy = (image - C11) ** 2 * Hphi[0] * Hphi[1]
    + (image - C10) ** 2 * Hphi[0] * Hinv[1]
    + (image - C01) ** 2 * Hinv[0] * Hphi[1]
    + (image - C00) ** 2 * Hinv[0] * Hinv[1]
    lenenergy = _curvature_energy(phi_arr)
    energy = np.sum(avgenergy) + np.sum(nu * lenenergy)

    return energy


def multiphase_chanvese_display(image, nu, max_iter=200, dt=0.1, fn=2, init_levelset=None):
    """
    multiphase chanvese模型,（用于展示），会返回更多的信息
    :param image:
    :param nu:
    :param tol:
    :param max_iter:
    :param dt:
    :param fn:
    :param init_levelset:
    :return:
    """
    if len(image.shape) != 2:
        raise ValueError("Image should be 2D array.")

    if init_levelset is None:
        phi_arr = _large_disk(image.shape, fn)
    else:
        phi_arr = init_levelset

    # 图像预处理
    image = image - np.min(image)
    if np.max(image) != 0:
        image = image / np.max(image)

    iter = 0
    segmentation = np.zeros(phi_arr.shape)
    segmentation[np.where(phi_arr > 0)] = 1
    seg_arr = [segmentation]

    while iter < max_iter:
        new_phi_arr = _calculate_phi(image, phi_arr, nu, fn, dt)
        phi_arr = new_phi_arr
        iter += 1

        # display
        if iter % 100 == 0:
            segmentation = np.zeros(phi_arr.shape)
            segmentation[np.where(phi_arr > 0)] = 1
            seg_arr.append(segmentation)

    segmentation = np.zeros(phi_arr.shape)
    segmentation[np.where(phi_arr > 0)] = 1
    return segmentation, phi_arr, seg_arr


def multiphase_chanvese(image, nu, max_iter=200, dt=0.1, fn=2, init_levelset=None):
    """
    multiphase chanvese 模型
    :param image:
    :param nu:
    :param max_iter:
    :param dt:
    :param fn:
    :param init_levelset:
    :return:
    """
    if len(image.shape) != 2:
        raise ValueError("Image should be 2D array.")

    if init_levelset is None:
        phi_arr = _large_disk(image.shape, fn)
    else:
        phi_arr = init_levelset

    image = image - np.min(image)
    if np.max(image) != 0:
        image = image / np.max(image)
    iter = 0
    while iter < max_iter:
        new_phi_arr = _calculate_phi(image, phi_arr, nu, fn, dt)
        segmentation = np.zeros(phi_arr.shape)
        segmentation[np.where(phi_arr > 0)] = 1
        phi_arr = new_phi_arr
        iter += 1

    return segmentation, phi_arr, []

