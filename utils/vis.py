from scipy import ndimage
from matplotlib import pyplot as plt
import seaborn as sns
import cv2

from utils.params import *


# Sample some images from the dataset and show them in a grid
def vis_dataset(nrows = 5, ncols = 5, add_masks = False):
    f, ax = plt.subplots(nrows = nrows, ncols = ncols, sharex = True, sharey = True, figsize=(20,20))
    sampled_imgs = np.random.choice(train_masks['img'], nrows*ncols)
#     sampled_imgs = [TRAIN_PATH + '/' + i for i in sampled_imgs]
    
    counter = 0
    for i in range(nrows):
        for j in range(ncols):
            car_code, angle_code = filename_to_code(sampled_imgs[counter])
            image = read_image(car_code, angle_code)
            ax[i, j].imshow(image)
            
            if add_masks:
                mask = read_image(car_code, angle_code, True)
#                 mix = cv2.bitwise_and(image, image, mask = mask)                    
                ax[i, j].imshow(mask, alpha = 0.4)
#                 ax[i, j].imshow(mix, cmap = 'Greys_r', alpha = 0.6)
            counter += 1
    plt.show()