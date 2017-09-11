
"""
    Some global parameters and constants used throughout different files and utilities.
"""

PROJECT_PATH = '.'
INPUT_PATH = PROJECT_PATH + '/input'

METADATA_CSV_PATH = INPUT_PATH + '/metadata.csv'

TRAIN_MASKS_CSV_PATH = INPUT_PATH + '/train_masks.csv'
TRAIN_PATH = INPUT_PATH + '/train'
TRAIN_MASKS_PATH = INPUT_PATH + '/train_masks'

SAMPLE_SUBMISSION_PATH = INPUT_PATH + '/sample_submission.csv'

TEST_DIR_PATH = INPUT_PATH + '/test'

INPUT_SIZE = 128
BATCH_SIZE = 16
MAX_EPOCHS = 100
THRESHOLD = 0.001