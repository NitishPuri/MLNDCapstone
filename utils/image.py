from scipy import ndimage
from matplotlib import pyplot as plt
import seaborn as sns
import cv2

from utils.params import *
from utils.filename import get_filepath_from_code

def read_image(car_code, angle_code, mask = False, test = False):
    img_path = get_filepath_from_code(car_code, angle_code, mask, test)
    img = None
    if mask is True:
        img = ndimage.imread(img_path, mode = 'L')
#         img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
#         img = cv2.resize(img, (INPUT_SIZE, INPUT_SIZE))
        img[img <= 127] = 0
        img[img > 127] = 1        
    else :
        img = ndimage.imread(img_path)
#         img = cv2.imread(img_path)
                
    return img
        
def resize(image):
    img = cv2.resize(image, (INPUT_SIZE, INPUT_SIZE))
    return img
    
def show_image(car_code, angle_code, mask = False):
    car_img = read_image(car_code, angle_code, mask)    
    plt.imshow(car_img)
    plt.show()

