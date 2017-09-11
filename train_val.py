from keras.callbacks import (CSVLogger, EarlyStopping, ModelCheckpoint,
                             ReduceLROnPlateau, TensorBoard)
from keras_tqdm import TQDMCallback, TQDMNotebookCallback
from sklearn.model_selection import train_test_split

import utils.data as data
import utils.generator as gen
import utils.models as models
import utils.zf_baseline as zf_baseline 
from utils.params import *

train_masks = data.read_train_masks()
train_images, validation_images = train_test_split(train_masks['img'], train_size = 0.8, random_state = 42)

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

def trainBaselineModel():
    baseline_model = models.get_baseline_model()

    callbacks = [ModelCheckpoint(filepath='models/baseline_model.best_weights.hdf5',
                        monitor = 'val_loss', verbose=2, save_best_only=True),
                TensorBoard(log_dir='./logs/baseline', histogram_freq = 1,
                batch_size = BATCH_SIZE, write_graph=True,
                write_images=True, write_grads=True),
                CSVLogger('./logs/baseline.csv'),
                EarlyStopping(monitor='val_loss', patience=8, verbose=1, min_delta=1e-4),
                ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=4, verbose=1, epsilon=1e-4),
                TQDMCallback()]

    baseline_history = baseline_model.fit_generator(generator=train_generator(),
                        steps_per_epoch= int(np.ceil(float(len(train_images)) / float(BATCH_SIZE))) , verbose = 0,
                        epochs = 100, validation_steps=int(np.ceil(float(len(validation_images)) / float(BATCH_SIZE))),
                        validation_data=valid_generator(), callbacks = callbacks)

    return baseline_history

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

def score_baseline_val_score():
    thr = 0.395     # Threshold calculated while generating mask image
    score = zf_baseline.get_score(validation_images, avg_mask, thr)

    print("Avg Mask Basline score on Validation Set : {:.6f}".format(score))
    input("\nPress Enter to continue...")