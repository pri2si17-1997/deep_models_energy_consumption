import os
import numpy as np
from skimage.io import imread

data_path = os.path.abspath('../dataset/np_data')

# train data
img_train_path = os.path.join(data_path, 'imgs_train.npy')
img_train_mask_path = os.path.join(data_path, 'imgs_mask_train.npy')
img_train_patients = os.path.join(data_path, 'imgs_patient.npy')
img_nerve_presence = os.path.join(data_path, 'nerve_presence.npy')

# test data
img_test_path = os.path.join(data_path, 'imgs_test.npy')
img_test_id_path = os.path.join(data_path, 'imgs_id_test.npy')

# image dimensions
image_rows = 420
image_cols = 580


# ======================================================================================================================
# Functions for test and train data creation, storage and access
# ======================================================================================================================

def load_test_data():
    """Load test data from a .npy file.

    :return: np.array with test data.
    """
    print('Loading test data from %s' % img_test_path)
    imgs_test = np.load(img_test_path)
    return imgs_test


def load_test_ids():
    """Load test ids from a .npy file.

    :return: np.array with test ids. Shape (samples, ).
    """
    print('Loading test ids from %s' % img_test_id_path)
    imgs_id = np.load(img_test_id_path)
    return imgs_id


def load_train_data():
    """Load train data from a .npy file.

    :return: np.array with train data.
    """
    print('Loading train data from %s and %s' % (img_train_path, img_train_mask_path))
    imgs_train = np.load(img_train_path)
    imgs_mask_train = np.load(img_train_mask_path)
    return imgs_train, imgs_mask_train


def load_patient_num():
    """Load the array with patient numbers from a .npy file

    :return: np.array with patient numbers
    """
    print('Loading patient numbers from %s' % img_train_patients)
    return np.load(img_train_patients)


def load_nerve_presence():
    """Load the array with binary nerve presence from a .npy file

    :return: np.array with patient numbers
    """
    print('Loading nerve presence array from %s' % img_nerve_presence)
    return np.load(img_nerve_presence)


def get_patient_nums(string):
    """Create a tuple (patient, photo) from image-file name patient_photo.tif

    :param string: image-file name in string format: patient_photo.tif
    :return: a tuple (patient, photo)

    >>> get_patient_nums('32_50.tif')
    (32, 50)
    """
    patient, photo = string.split('_')
    photo = photo.split('.')[0]
    return int(patient), int(photo)


def get_nerve_presence(mask_array):
    """Create an array specifying nerve presence on each of the masks in the mask_array

    :param mask_array: 4D tensor of a shape (samples, rows, cols, channels=1) with masks
    :return:
    """
    print("type(mask_array):", type(mask_array))
    print("mask_array.shape:", mask_array.shape)
    return np.array([int(np.sum(mask_array[i, :, :, 0]) > 0) for i in range(mask_array.shape[0])])