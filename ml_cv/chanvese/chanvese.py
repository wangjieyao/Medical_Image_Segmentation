import numpy as np
from scipy.ndimage import distance_transform_edt as distance

def _large_disk(image_size):
    """
    Generate a disk level set function
    生成一个圆形的水平集函数
    :param image_size:
    :return:
    """
    res = np.ones(image_size)
    centerY = int((image_size[0]-1) / 2)
    centerX = int((image_size[1]-1) / 2)
    res[centerY, centerX] = 0.
    radius = float(min(centerX, centerY))
    return (radius-distance(res)) / radius

def _heavyside(x, eps=1.):
    """
    Returns the result of a regularised heavyside function of the input value.
    返回Heavyside函数（单位越阶函数）结果，它会逼近于 0.5 + 1/π * arctan(kx)
    :param x:
    :param eps:
    :return:
    """
    # 下面两个函数结果差不多
    # return 0.5 + (1. / np.pi) * np.arctan(x / eps)
    return 1 * (x > 0)

def _delta(x, eps=1.):
    return eps / (eps ** 2 + x ** 2)

def _curvature_energy(phi):
    """
    Returns the result of a regularised dirac function of the input value.
    返回狄拉克δ函数的结果
    :param phi:
    :return:
    """
    P = np.pad(phi, 1, mode='edge')
    fy = (P[2:, 1:-1] - P[:-2, 1:-1]) / 2.0
    fx = (P[1:-1, 2:] - P[1:-1, :-2]) / 2.0
    fyy = P[2:, 1:-1] + P[:-2, 1:-1] - 2 * phi
    fxx = P[1:-1, 2:] + P[1:-1, :-2] - 2 * phi
    fxy = .25 * (P[2:, 2:] + P[:-2, :-2] - P[:-2, 2:] - P[2:, :-2])
    grad2 = fx ** 2 + fy ** 2
    K = ((fxx * fy ** 2 - 2 * fxy * fx * fy + fyy * fx ** 2) /
         (grad2 * np.sqrt(grad2) + 1e-8))
    return K

def _curvature(phi):
    eta = 1e-16
    P = np.pad(phi, 1, mode='edge')
    phixp = P[1:-1, 2:] - P[1:-1, 1:-1]
    phixn = P[1:-1, 1:-1] - P[1:-1, :-2]
    phix0 = (P[1:-1, 2:] - P[1:-1, :-2]) / 2.0
    phiyp = P[2:, 1:-1] - P[1:-1, 1:-1]
    phiyn = P[1:-1, 1:-1] - P[:-2, 1:-1]
    phiy0 = (P[2:, 1:-1] - P[:-2, 1:-1]) / 2.0
    C1 = 1. / np.sqrt(eta + phixp ** 2 + phiy0 ** 2)
    C2 = 1. / np.sqrt(eta + phixn ** 2 + phiy0 ** 2)
    C3 = 1. / np.sqrt(eta + phix0 ** 2 + phiyp ** 2)
    C4 = 1. / np.sqrt(eta + phix0 ** 2 + phiyn ** 2)
    K = (P[1:-1, 2:] * C1 + P[1:-1, :-2] * C2 +
         P[2:, 1:-1] * C3 + P[:-2, 1:-1] * C4)
    return K, (C1 + C2 + C3 + C4)

def _calculate_averages(image, Hphi):
    """
    Calculate the average pixels 'inside' and 'outside'.
    计算分割内外的平均像素
    :param image:
    :param Hphi:
    :return:
    """
    Hinv = 1. - Hphi
    Hphi_sum = np.sum(Hphi)
    Hinv_sum = np.sum(Hinv)
    avg_inside = np.sum(image * Hphi)
    avg_oustide = np.sum(image * Hinv)
    if Hphi_sum != 0:
        avg_inside /= Hphi_sum
    if Hinv_sum != 0:
        avg_oustide /= Hinv_sum
    return avg_inside, avg_oustide

def _calculate_phi(image, phi, mu, lambda1, lambda2, dt):
    """
    Returns the variation of level set 'phi' based on algorithm parameters.
    返回水平集函数中的phi变量
    :param image:
    :param phi:
    :param mu:
    :param lambda1:
    :param lambda2:
    :param dt:
    :return:
    """
    K, C_sum = _curvature(phi)
    Hphi = _heavyside(phi)
    c1, c2 = _calculate_averages(image, Hphi)

    difference_from_average_term = (- lambda1 * (image - c1) ** 2 +
                                    lambda2 * (image - c2) ** 2)
    new_phi = (phi + (dt * _delta(phi)) *
               (mu * K + difference_from_average_term))
    return new_phi / (1 + mu * dt * _delta(phi) * C_sum)

def _energy(image, phi, mu, lambda1, lambda2):
    """
    Returns the total 'energy' of the current level set function.
    返回当前水平集函数中的总能量
    :param image:
    :param phi:
    :param mu:
    :param lambda1:
    :param lambda2:
    :return:
    """
    Hphi = _heavyside(phi)
    (c1, c2) = _calculate_averages(image, Hphi)
    Hinv = 1. - Hphi
    avgenergy = (lambda1 * (image-c1)**2 * Hphi + lambda2 * (image-c2)**2 * Hinv)
    lenenergy = _curvature_energy(phi)
    if np.sum(lenenergy) < 10:
        _curvature_energy(phi)
    return np.sum(avgenergy) + np.sum(mu * lenenergy)

def chan_vese(image, mu=0.25, lambda1=1, lambda2=1, tol=1e-3, max_iter=200,
              dt=0.5, init_levelset = None):
    """
    chanvese模型
    :param image:
    :param mu:
    :param lambda1:
    :param lambda2:
    :param tol:
    :param max_iter:
    :param dt:
    :param init_levelset:
    :return:
    """
    if len(image.shape) != 2:
        raise ValueError("Image should be 2D array.")
    if init_levelset is None:
        phi = _large_disk(image.shape)
    else:
        phi = init_levelset
    image = image - np.min(image)
    if np.max(image) != 0:
        image = image / np.max(image)
    segmentation = np.zeros(phi.shape)
    segmentation[np.where(phi > 0)] = 1
    seg_arr = []
    iter = 0
    while iter < max_iter:
        new_phi = _calculate_phi(image, phi, mu, lambda1, lambda2, dt)
        phi = new_phi
        iter += 1
    segmentation = np.zeros(phi.shape)
    segmentation[np.where(phi > 0)] = 1
    return segmentation, phi, seg_arr

