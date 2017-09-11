import numpy as np
import keras.backend as K
from keras.losses import binary_crossentropy

# Evaluation Metric: Dice Coefficient
# Given two vector x and y, returns their dice distance

"""
    Accuracy and loss functions for segmentation model. Defines Dice accuracy and loss function.
"""

def dice(x, y):
    return 2*(len(set(x).intersection(set(y))))/(len(set(x)) + len(set(y)))    

def dice_coeff_numpy(im1, im2, empty_score=1.0):
    im1 = im1.astype(np.bool)
    im2 = im2.astype(np.bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    im_sum = im1.sum() + im2.sum()
    if im_sum == 0:
        return empty_score

    intersection = np.logical_and(im1, im2)
    return 2. * intersection.sum() / im_sum

def dice_coeff(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return score

def dice_loss(y_true, y_pred):
    return (1 - dice_coeff(y_true, y_pred))

def bce_dice_loss(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
