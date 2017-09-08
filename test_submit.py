import numpy as np
import pandas as pd
from tqdm import tqdm

import utils.data as data
import utils.encoder as encoder
import utils.models as models
from utils.filename import *
from utils.image import *
from utils.params import *

# train_masks = pd.read_csv(TRAIN_MASKS_CSV_PATH)
# print(train_masks.head())

# encoder.test_rle_encode(train_masks)


baseline_model = models.get_baseline_model()
# baseline_model.summary()

# pred = baseline_model.predict()

# sample_car_code = '00087a6bd4dc'
# sample_angle_code = '04'

# fig, 


threshold = 0.001
rles = []

orig_width = 1918
orig_height = 1280

ids_test = data.list_test_files()
print('Predicting on {} samples with batch_size = {}...'.format(len(ids_test), BATCH_SIZE))
for start in tqdm(range(0, len(ids_test), BATCH_SIZE)):
    x_batch = []
    end = min(start + BATCH_SIZE, len(ids_test))
    ids_test_batch = ids_test[start:end]
    for id in ids_test_batch:
        car_code, angle_code = filename_to_code(id)
        img = read_image(car_code, angle_code, test = True)
        img = resize(img)
        x_batch.append(img)
    x_batch = np.array(x_batch, np.float32) / 255
    preds = baseline_model.predict_on_batch(x_batch)
    preds = np.squeeze(preds, axis = 3)
    for pred in preds:
        prob = cv2.resize(pred, (orig_width, orig_height))
        mask = prob > threshold
        rle = encoder.run_length_encode(mask)
        rles.append(rle)

print("Generating submission file,..")
df = pd.DataFrame({'img':names, 'rle_mask':rles})
df.to_csv('submit/baseline_{}.csv.gz'.format(threshold), index = False, compression='gzip')
