import numpy as np
from scipy.ndimage import distance_transform_edt as distance

from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
from skimage import measure, draw
from skimage import transform, data
import cv2
import time
import nibabel as nib
from skimage import data, img_as_float
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from itertools import chain


KIDNEY_PATH = '/Users/WangJieYao/Mycode/MSC/Medical_Image_Segmentation/image/qubib/kidney/Training/case{0}/image.nii.gz'
KIDNEY_S_PATH = '/Users/WangJieYao/Mycode/MSC/Medical_Image_Segmentation/image/qubib/kidney/Training/case{0}/task01_seg01.nii.gz'

def load_lung(path='/Users/WangJieYao/Mycode/MSC/ChanVese_ML/image/scr/All247images/JPCLN001.IMG'):
    SEG_PATH_LEFTLUNG = "/Users/WangJieYao/Mycode/MSC/ChanVese_ML/image/scr/scratch/fold{0}/masks/left lung/{1}.gif"
    SEG_PATH_RIGHTLUNG = "/Users/WangJieYao/Mycode/MSC/ChanVese_ML/image/scr/scratch/fold{0}/masks/right lung/{1}.gif"

    shape = (2048, 2048)  # matrix size
    dtype = np.dtype('>u2')  # big-endian unsigned integer (16bit)
    image = np.fromfile(open(path, 'rb'), dtype).reshape(shape)
    image = transform.resize(image, (256,256), preserve_range=True)

    name = path[-12:-4]
    num = 2 if int(path[-5]) % 2 == 0 else 1
    seg_l_path = SEG_PATH_LEFTLUNG.format(num, name)
    seg_r_path = SEG_PATH_RIGHTLUNG.format(num, name)
    l_img = Image.open(seg_l_path)
    r_img = Image.open(seg_r_path)
    plt.figure()
    r_img = np.asarray(r_img).copy()
    r_img[r_img > 0] = 1
    l_img = np.asarray(l_img).copy()
    l_img[l_img > 0] = 2
    l_img = transform.resize(l_img,  (256,256), preserve_range=True)
    r_img = transform.resize(r_img,  (256,256), preserve_range=True)

    y_start = 10
    y_end = 220
    x_start = 25
    x_end = 230

    l_img = l_img[y_start:y_end, x_start:x_end]
    r_img = r_img[y_start:y_end, x_start:x_end]
    r_img[r_img > 0] = 1
    l_img[l_img > 0] = 1
    target = np.zeros((2,l_img.shape[0], l_img.shape[1]))
    target[0] = r_img
    target[1] = l_img
    l_img[l_img > 0] = 2
    label = l_img + r_img

    label = label.astype(int)
    image = image[y_start:y_end, x_start:x_end]

    return image, label, target


def load_prostate():
    file = '/Users/WangJieYao/Mycode/MSC/ChanVese_ML/image/qubib/prostate/Training/case04/image.nii.gz'
    # img = nib.load(file)
    # plt.imshow(img.dataobj[:, :, 0])
    seg_file = '/Users/WangJieYao/Mycode/MSC/ChanVese_ML/image/qubib/prostate/Training/case04/task01_seg01.nii.gz'
    # seg_img = nib.load(seg_file)
    # plt.figure()
    # plt.imshow(seg_img.dataobj[:, :, 0])
    # plt.show()

    img = nib.load(file)
    image = img_as_float(img.dataobj[:, :, 0])
    seg_img = nib.load(seg_file)
    seg_image = img_as_float(seg_img.dataobj[:, :, 0])



    image = image[180:550, 180:600]
    label = seg_image[180:550, 180:600]

    plt.figure()
    plt.imshow(image)
    plt.figure()
    plt.imshow(label)
    plt.show()

    x = np.array(list(chain(*image)))
    y = np.array(list(chain(*label)))
    # x = x.reshape((-1,1))

    front_i = set(np.where(x > 0)[0])
    others_i = front_i - set((np.where(y > 0))[0])
    y[list(others_i)] = 0

    plt.figure()
    plt.imshow(y.reshape(image.shape))
    plt.show()
    return image, label,


def load_lungs(img_numbers = 5, display = False):
    PATH='/Users/WangJieYao/Mycode/MSC/Medical_Image_Segmentation/image/scr/All247images/JPCLN{0}.IMG'
    SEG_PATH_LEFTLUNG = "/Users/WangJieYao/Mycode/MSC/Medical_Image_Segmentation/image/scr/scratch/fold{0}/masks/left lung/JPCLN{1}.gif"
    SEG_PATH_RIGHTLUNG = "/Users/WangJieYao/Mycode/MSC/Medical_Image_Segmentation/image/scr/scratch/fold{0}/masks/right lung/JPCLN{1}.gif"


    resize_shape =  (256,256)
    shape = (2048, 2048)  # matrix size
    if display:

        y_start = 10
        y_end = 230
        x_start = 20
        x_end = 235
    else:
        y_start = 10
        y_end = 220
        x_start = 25
        x_end = 230
    image_shape = (y_end - y_start, x_end - x_start)

    images = np.empty((img_numbers, image_shape[0], image_shape[1]))
    targets = np.empty((img_numbers, image_shape[0], image_shape[1]))
    targets_separate = np.empty((img_numbers, 2, image_shape[0], image_shape[1]))
    centers = np.empty((img_numbers, 2, 2))

    for i in range(img_numbers):
        name = str(i+1).zfill(3)
        num = 2 if (i+1) % 2 == 0 else 1
        path = PATH.format(name)
        seg_l_path = SEG_PATH_LEFTLUNG.format(num, name)
        seg_r_path = SEG_PATH_RIGHTLUNG.format(num, name)

        # open and resize image
        dtype = np.dtype('>u2')  # big-endian unsigned integer (16bit)
        image = np.fromfile(open(path, 'rb'), dtype).reshape(shape)
        image = transform.resize(image, resize_shape, preserve_range=True)

        l_img = Image.open(seg_l_path)
        r_img = Image.open(seg_r_path)
        r_img = np.asarray(r_img).copy()
        l_img = np.asarray(l_img).copy()
        l_img = transform.resize(l_img, resize_shape, preserve_range=True)
        r_img = transform.resize(r_img, resize_shape, preserve_range=True)
        r_img[r_img > 0] = 1
        l_img[l_img > 0] = 2

        # crop image
        l_img = l_img[y_start:y_end, x_start:x_end]
        r_img = r_img[y_start:y_end, x_start:x_end]
        images[i] = image[y_start:y_end, x_start:x_end]
        targets[i] = (l_img + r_img).astype(int)
        targets_separate[i,0] = r_img
        l_img[l_img > 0] = 1
        targets_separate[i,1] = l_img

        # find certer
        centers[i, 0] = find_certer(r_img)
        centers[i, 1] = find_certer(l_img)

    return images, targets, targets_separate, centers

def find_certer(image):
    r = np.where(image>0)[0]
    c = np.where(image>0)[1]
    center_y = int((r.max() - r.min()) / 2 + r.min())
    center_x = int((c.max() - c.min()) / 2 + c.min())
    return np.array([center_x, center_y])


def load_tumor():
    file = '/Users/WangJieYao/Mycode/MSC/ChanVese_ML/image/qubib/brain-tumor/Training/case03/image.nii.gz'
    seg_file = '/Users/WangJieYao/Mycode/MSC/ChanVese_ML/image/qubib/brain-tumor/Training/case03/task01_seg01.nii.gz'

    img = nib.load(file)
    image = img_as_float(img.dataobj[:, :, 0])
    seg_img = nib.load(seg_file)
    seg_image = img_as_float(seg_img.dataobj)

    image = image[75:175, 60:140].copy()
    seg_image = seg_image[75:175, 60:140].copy()



    plt.figure()
    plt.imshow(image,  plt.cm.gray)
    plt.figure()
    plt.imshow(seg_image)
    plt.show()
    return image, seg_image


def load_kidneys(img_numbers):
    row = 125
    col = 150
    images = np.empty((img_numbers, row, col))
    segs = np.empty((img_numbers, row, col))
    centers = np.empty((img_numbers, 2))
    for n in range(img_numbers):
        i = str(n+1).zfill(2)
        img = nib.load(KIDNEY_PATH.format(i))
        image = img_as_float(img.dataobj)
        seg_img = nib.load(KIDNEY_S_PATH.format(i))
        seg_image = img_as_float(seg_img.dataobj)
        r = np.where(seg_image>0)[0]
        c = np.where(seg_image>0)[1]
        row_s = r.min() - 20
        col_s = c.min() - 20
        center_y = int((r.max() - r.min()) / 2 + 20)
        center_x = int((c.max() - c.min()) / 2 + 20)
        images[n] = image[row_s:row_s + row, col_s: col_s + col]
        segs[n] = seg_image[row_s:row_s + row, col_s: col_s + col]
        centers[n] = np.array([center_x, center_y])


        # plt.figure()
        # plt.imshow(image, plt.cm.gray)
        # plt.figure()
        # plt.imshow(seg_image)
        # plt.figure()
        # plt.imshow(image[row_s:row_s + row, col_s: col_s + col], plt.cm.gray)
        # plt.show()
    return images, segs, centers




load_kidneys(10)