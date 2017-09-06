import numpy as np
import keras.backend as K
from keras.losses import binary_crossentropy

# Evaluation Metric: Dice Coefficient
# Given two vector x and y, returns their dice distance

laplace_smoothing = 1

def test(self):
    print("helloooo")
    pass

def dice(x, y):
    return 2*(len(set(x).intersection(set(y))))/(len(set(x)) + len(set(y)))    

def dice_coeff(im1, im2, empty_score=1.0):
    im1 = im1.astype(np.bool)
    im2 = im2.astype(np.bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    im_sum = im1.sum() + im2.sum()
    if im_sum == 0:
        return empty_score

    intersection = np.logical_and(im1, im2)
    return 2. * intersection.sum() / im_sum

def dice_loss(y_true, y_pred):
    return (1 - dice_coeff(y_true, y_pred))

def bce_dice_loss(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)

def get_score(train_masks, avg_mask, thr):
    d = 0.0
    for i in range(train_masks.shape[0]):
        d += dice_coeff(train_masks[i], avg_mask)
    return d/train_masks.shape[0]
