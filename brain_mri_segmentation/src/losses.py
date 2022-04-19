from keras.losses import binary_crossentropy
import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np

epsilon = 1e-5
smooth = 1

def tversky(y_true, y_pred):
    y_true_pos = K.flatten(y_true)
    y_pred_pos = K.flatten(y_pred)
    true_pos = K.sum(y_true_pos * y_pred_pos)
    false_neg = K.sum(y_true_pos * (1-y_pred_pos))
    false_pos = K.sum((1-y_true_pos)*y_pred_pos)
    alpha = 0.7
    return (true_pos + smooth)/(true_pos + alpha*false_neg + (1-alpha)*false_pos + smooth)

def focal_tversky(y_true,y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    pt_1 = tversky(y_true, y_pred)
    gamma = 0.75
    return K.pow((1-pt_1), gamma)

def tversky_loss(y_true, y_pred):
    return 1 - tversky(y_true,y_pred)

def dice_coef(mask_1, mask_2, smooth=1):
    """Compute the dice coefficient between two equal-sized masks.

    Dice Coefficient: https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient
    We need to add smooth, because otherwise 2 empty (all zeros) masks will throw an error instead of
    giving 1 as an output.

    :param mask_1: first mask
    :param mask_2: second mask
    :param smooth: Smoothing parameter for dice coefficient
    :return: Smoothened dice coefficient between two equal-sized masks
    """
    mask_1_flat = K.flatten(mask_1)
    mask_2_flat = K.flatten(mask_2)

    # for pixel values in {0, 1} multiplication is the intersection of masks
    intersection = K.sum(mask_1_flat * mask_2_flat)
    return (2. * intersection + smooth) / (K.sum(mask_1_flat) + K.sum(mask_2_flat) + smooth)


def dice_coef_loss(mask_pred, mask_true):
    """Calculate dice coefficient loss, when comparing predicted mask for an image with the true mask

    :param mask_pred: predicted mask
    :param mask_true: true mask
    :return: dice coefficient loss
    """
    return -dice_coef(mask_pred, mask_true)


def np_dice_coef(mask_1, mask_2, smooth=1):
    """Compute the dice coefficient between two equal-sized masks.

    Used for testing on artificially generated np.arrays

    Dice Coefficient: https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient
    Need smooth, because otherwise 2 empty (all zeros) masks will throw an error instead of giving 1 as an output.

    :param mask_1: first mask
    :param mask_2: second mask
    :param smooth: Smoothing parameter for dice coefficient
    :return: Smoothened dice coefficient between two equal-sized masks
    """
    tr = mask_1.flatten()
    pr = mask_2.flatten()
    return (2. * np.sum(tr * pr) + smooth) / (np.sum(tr) + np.sum(pr) + smooth)