import os

import pandas as pd

from utils.params import *


def read_test_ids():
    """
        Read sample submission file, list and return all test image ids.
    """
    df_test = pd.read_csv(SAMPLE_SUBMISSION_PATH)
    ids_test = df_test['img'].map(lambda s: s.split('.')[0])

    return ids_test

def read_train_masks():
    """
        Read and return `train_masks.csv`
    """
    train_masks = pd.read_csv(TRAIN_MASKS_CSV_PATH)
    return train_masks

def read_metadata():
    """
        Read and return `metadata.csv`
    """
    return pd.read_csv(METADATA_CSV_PATH)

def list_test_files():
    """
        List all test files from the `./input/test`
    """
    return os.listdir(TEST_DIR_PATH)


def list_car_and_dog_images():
    return ['./input/car_and_dog/' + file  for file in os.listdir('./input/car_and_dog')] 
