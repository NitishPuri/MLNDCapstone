import time
import numpy as np

from utils.filename import filename_to_code
from utils.image import read_image

def rle_encode(mask_image):
    """
        Modified from : https://www.kaggle.com/stainsby/fast-tested-rle
        Convert a mask image into a list of run lengths
    """
    pixels = mask_image.flatten()
    # We avoid issues with '1' at the start or end (at the corners of 
    # the original image) by setting those pixels to '0' explicitly.
    # We do not expect these to be non-zero for an accurate mask, 
    # so this should not harm the score.
    pixels[0] = 0
    pixels[-1] = 0
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 2
    runs[1::2] = runs[1::2] - runs[:-1:2]
    return runs

def rle_to_string(runs):
    return ' '.join(str(x) for x in runs)

def run_length_encode(mask):
    """
        Convert a mask image into its run length encoding
    """
    return rle_to_string(rle_encode(mask))

def test_rle_encode(train_masks):
    """
        Modified from : https://www.kaggle.com/stainsby/fast-tested-rle
    """
    test_mask = np.asarray([[0, 0, 0, 0], [0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 0, 0]])
    assert rle_to_string(rle_encode(test_mask)) == '7 2 11 2'
    num_masks = len(train_masks['img'])
    print('Verfiying RLE encoding on', num_masks, 'masks ...')
    time_read = 0.0 # seconds
    time_rle = 0.0 # seconds
    time_stringify = 0.0 # seconds
    for mask_idx in range(num_masks):
        img_file_name = train_masks.loc[mask_idx, 'img']
        car_code, angle_code = filename_to_code(img_file_name)
        t0 = time.clock()
        mask_image = read_image(car_code, angle_code, True)
        time_read += time.clock() - t0
        t0 = time.clock()
        rle_truth_str = train_masks.loc[mask_idx, 'rle_mask']
        rle = rle_encode(mask_image)
        time_rle += time.clock() - t0
        t0 = time.clock()
        rle_str = rle_to_string(rle)
        time_stringify += time.clock() - t0
        assert rle_str == rle_truth_str
        if mask_idx and (mask_idx % 500) == 0:
            print('  ..', mask_idx, 'tested ..')
    print('Time spent reading mask images:', time_read, 's, =>', \
            1000*(time_read/num_masks), 'ms per mask.')
    print('Time spent RLE encoding masks:', time_rle, 's, =>', \
            1000*(time_rle/num_masks), 'ms per mask.')
    print('Time spent stringifying RLEs:', time_stringify, 's, =>', \
            1000*(time_stringify/num_masks), 'ms per mask.')

