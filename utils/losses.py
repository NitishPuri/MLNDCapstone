import numpy as np
import keras.backend as K
from keras.losses import binary_crossentropy

# Evaluation Metric: Dice Coefficient
# Given two vector x and y, returns their dice distance

laplace_smoothing = 0

def dice(x, y):
    return 2*(len(set(x).intersection(set(y))))/(len(set(x)) + len(set(y)))    

def dice_coeff(y_true, y_pred):
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f*y_pred_f)
        return (2. * intersection + laplace_smoothing)/(K.sum(y_true_f) + K.sum(y_pred_f) + 2*laplace_smoothing)

def dice_loss(y_true, y_pred):
    return (1 - dice_coeff(y_true, y_pred))

def bce_dice_loss(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)