import cv2
import numpy as np
from keras.callbacks import (CSVLogger, EarlyStopping, ModelCheckpoint,
                             ReduceLROnPlateau, TensorBoard)
from keras_tqdm import TQDMCallback, TQDMNotebookCallback
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import utils.data as data
import utils.generator as gen
import utils.losses as losses
import utils.models as models
import utils.zf_baseline as zf_baseline
from utils.filename import *
from utils.image import *
from utils.params import *

train_masks = data.read_train_masks()
train_images, validation_images = train_test_split(train_masks['img'], train_size = 0.8, random_state = 42)

""" Manufacturer Model """
def trainManufacturerModel():
    manufacturer_model = models.get_manufacturer_model()

    callbacks = [ ModelCheckpoint(filepath='models/manufacturer_model.best_weights.hdf5', 
                                verbose=2, save_best_only=True),
                  TensorBoard(log_dir='./logs/man', histogram_freq = 2),
                  TQDMCallback(),   ## Add callback for console
            #   TQDMNotebookCallback(),   ## Add callback for Notebook
                  CSVLogger('./logs/man.log') ]

    manufacturer_model.summary()

    manufacturer_model.fit_generator(train_manufacturer_gen(), steps_per_epoch=len(train_masks),
                                        verbose = 0, epochs = 10, validation_steps=100,
                                        validation_data=gen.val_manufacturer_gen(), 
                                        callbacks = callbacks)

def show_manufacturerModel_summary():
    manufacturer_model = models.get_manufacturer_model().summary()
    input("Press Enter to continue...")


""" Basic Basline Model """

def trainBaselineModel():
    baseline_model = models.get_baseline_model()

    callbacks = [ModelCheckpoint(filepath='models/baseline_model.best_weights.hdf5',
                        monitor = 'val_loss', verbose=2, save_best_only=True),
                # TensorBoard(log_dir='./logs/baseline', histogram_freq = 1,
                #             batch_size = BATCH_SIZE, write_graph=True,
                #             write_images=True, write_grads=True),
                CSVLogger('./logs/baseline.csv'),
                EarlyStopping(monitor='val_loss', patience=8, verbose=1, min_delta=1e-4),
                ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=4, verbose=1, epsilon=1e-4),
                TQDMCallback()]

    baseline_history = baseline_model.fit_generator(generator=gen.train_generator(),
                        steps_per_epoch= int(np.ceil(float(len(train_images)) / float(BATCH_SIZE))) , verbose = 0,
                        epochs = 100, validation_steps=int(np.ceil(float(len(validation_images)) / float(BATCH_SIZE))),
                        validation_data=gen.valid_generator(), callbacks = callbacks)

    print("Training complete,..")
    input("Press Enter to continue,..")

    return baseline_history

def show_baseline_2_summary():
    models.get_baseline_model().summary()
    input("Press Enter to continue...")

""" Avg Mask Baseline Model"""

def score_baseline_val_score():
    num_val_samples = 600
    val_batch = validation_images[0:num_val_samples]                 

    avg_mask = cv2.imread('images/avg_mask.jpg', cv2.IMREAD_GRAYSCALE)

    d = 0.0
    
    print("Calculating score over {} validation images.".format(num_val_samples))

    for val_img in tqdm(val_batch):
        car_code, angle_code = filename_to_code(val_img)
        mask = read_image(car_code, angle_code, mask=True)
        d += losses.dice_coeff_numpy(mask, avg_mask)
        # d += losses.dice_coeff(mask, avg_mask)

    d = d/num_val_samples

    print("Avg Mask Basline score on Validation Set : {:.6f}".format(d))
    input("\nPress Enter to continue...")

def show_avg_mask():
    img = cv2.imread('images/avg_mask.jpg')
    if img is None:
        print("Creating avg mask,..")
        score, img  = zf_baseline.validation_get_optimal_thr()
    
    cv2.namedWindow('Avg Mask', cv2.WINDOW_NORMAL)
    cv2.imshow('Avg Mask', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def create_avgMask_submission():
    img = cv2.imread('images/avg_mask.jpg')
    if img is None:
        print("Creating avg mask,..")
        score, img  = zf_baseline.validation_get_optimal_thr()
    zf_baseline.create_submission(img)


def trainUnet128Model():
    unet_model = models.get_unet_128()

    callbacks = [ModelCheckpoint(filepath='models/unet_128.best_weights.hdf5',
                                 monitor = 'val_loss', verbose=2, save_best_only=True),
                # TensorBoard(log_dir='./logs/unet_128', histogram_freq = 1,
                #             batch_size = BATCH_SIZE, write_graph=True,
                #             write_images=True, write_grads=True),
                CSVLogger('./logs/unet_128_history01.csv'),
                EarlyStopping(monitor='val_loss', patience=8, verbose=1, min_delta=1e-4),
                ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=4, verbose=1, epsilon=1e-4),
                TQDMCallback()]

    steps_per_epoch = int(np.ceil(float(len(train_images)) / float(BATCH_SIZE)))
    validation_steps = int(np.ceil(float(len(validation_images)) / float(BATCH_SIZE)))
    
    unet_model_history = unet_model.fit_generator(generator = train_generator(),
                        steps_per_epoch = steps_per_epoch, verbose = 0,
                        epochs = 100, validation_steps = validation_steps,
                        validation_data=valid_generator(), callbacks = callbacks)

    return unet_model_history

def show_uNet_summary():
    models.get_unet_128().summary()
    input("Press Enter to continue...")
