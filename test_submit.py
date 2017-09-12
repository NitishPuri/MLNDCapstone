import numpy as np
import pandas as pd
from tqdm import tqdm

import utils.data as data
import utils.encoder as encoder
import utils.models as models
from utils.filename import *
from utils.image import *
from utils.params import *

import scipy.misc


baseline_model = models.get_baseline_model()
baseline_model.name = 'baseline'
baseline_model.load_weights('models/baseline_model.best_weights.hdf5')

unet_128_model = models.get_unet_128()
unet_128_model.name = 'unet_128'
unet_128_model.load_weights('models/unet_128.best_weights.hdf5')

samples = data.list_car_and_dog_images()


modelList = [baseline_model, unet_128_model]
for i, model in enumerate(modelList):
    for sample in tqdm(samples):
        image = ndimage.imread(sample)

        f, ax = plt.subplots(nrows = 1, ncols = 3, sharex = True, sharey = True, figsize=(20,7))
        
        # print(image.shape)
        # try:
        image = image[:,:,:3]

        im = resize(image)
        
        x_batch = []
        x_batch.append(im)
        x_batch = np.array(x_batch, np.float32) /255
        pred = model.predict(x_batch).squeeze()

        im = image
        pred = cv2.resize(pred, (image.shape[1], image.shape[0]))

        filename = sample[0:-4]

        ax[0].imshow(im)
        ax[0].set_title('input')
        ax[1].imshow(pred, cmap='gray')
        ax[1].set_title('output')
        ax[2].imshow(pred > THRESHOLD, cmap='gray')
        ax[2].set_title('threshold mask')

        # scipy.misc.imsave('{}_{}_out.jpg'.format(filename, i), pred)
        # scipy.misc.imsave('{}_{}_mask.jpg'.format(filename, i), (pred > THRESHOLD))        

        filename = filename.split('/')[-1]
        f.suptitle('{} , {}'.format(model.name, filename))
        f.savefig('images/car_dog_out/{}_{}.png'.format(model.name, filename, ), dpi = f.dpi)
        plt.close(f)

        # except IndexError:
        #     print("Issue with '{}'".format(sample))
        #     print(ndimage.imread(sample).shape)
            
