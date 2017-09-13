import cv2
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy import ndimage
import scipy.misc
from tqdm import tqdm

import utils.data as data
from utils.filename import *
from utils.image import *
from utils.params import *
from utils.preprocess import *
import utils.models as models

train_masks = data.read_train_masks()

""" Visualization utils"""

def vis_dataset(nrows = 5, ncols = 5, mask_alpha = 0.4, augment = False):
    
    """ Sample some images from the dataset and show them in a grid."""

    f, ax = plt.subplots(nrows = nrows, ncols = ncols, sharex = True, sharey = True, figsize=(20,20))
    sampled_imgs = np.random.choice(train_masks['img'], nrows*ncols)
    
    counter = 0
    for i in range(nrows):
        for j in range(ncols):
            car_code, angle_code = filename_to_code(sampled_imgs[counter])
            image = read_image(car_code, angle_code)
            mask = read_image(car_code, angle_code, True)

            if augment is True:
                image = resize(image)
                mask = resize(mask)
                image = randomHueSaturationVariation(image, hue_shift_limit=(-50,50),
                                                    sat_shift_limit=(-5, 5), val_shift_limit=(-15, 15))
                image, mask = randomShiftScaleRotate(image, mask,  rotate_limit=(-5, 5))
                image, mask = randomHorizontalFlip(image, mask)
                
            ax[i, j].imshow(image)
            ax[i, j].imshow(mask, alpha = mask_alpha)

            counter += 1
    plt.show()

# def vis_manufacturer_predictions(nrows = 5, ncols = 5):
    
#     """ Sample some images from the dataset and show them in a grid."""

#     model = models.get_manufacturer_model()
#     model.load_weights('./models/manufacturer_model.best_weights.hdf5')

#     f, ax = plt.subplots(nrows = nrows, ncols = ncols, sharex = True, sharey = True, figsize=(20,20))
#     sampled_imgs = np.random.choice(train_masks['img'], nrows*ncols)
    
#     counter = 0
#     for i in range(nrows):
#         for j in range(ncols):
#             car_code, angle_code = filename_to_code(sampled_imgs[counter])
#             image = read_image(car_code, angle_code)

#             image = resize(image)
                
#             x_batch = []
#             x_batch.append(image)
#             x_batch = np.array(x_batch, np.float32) /255
#             pred = model.predict(x_batch).squeeze()

#             ax[i, j].imshow(image)
#             model.predict

#             counter += 1
#     plt.show()

# Sample some images from the dataset and show them in a grid
def vis_curropted_dataset():

    """ List of incorrectly labeled images is given."""
    """ Sample and show some images from it."""

    curropted_masks =  ['0d1a9caf4350_14', '1e89e1af42e7_07', '2a4a8964ebf3_08', 
                        '2ea62c1beee7_03', '2faf504842df_03', '2faf504842df_12', 
                        '3afec4b5ac07_05', '3afec4b5ac07_12', '3afec4b5ac07_13', 
                        '3afec4b5ac07_14', '3bca821c8c41_13', '4a4364d7fc6d_06',
                        '4a4364d7fc6d_07', '4a4364d7fc6d_14', '4a4364d7fc6d_15', 
                        '4baf50a3d8c2_05', '4e5ac4b9f074_11', '4f1f065d78ac_14',
                        '4f0397cf7937_05', '5df60cf7cab2_07', '5df60cf7cab2_15', 
                        '6ba36af67cb0_07', '6bff9e10288e_01' ]

    ncols = 2
    nrows = 2

    curropted_masks = np.random.choice(curropted_masks, ncols*nrows)
    f, ax = plt.subplots(nrows = nrows, ncols = ncols, sharex = True, sharey = True, figsize=(20, 20))
    sampled_imgs = np.random.choice(train_masks['img'], nrows*ncols)
    
    counter = 0
    for i in range(nrows):
        for j in range(ncols):
            if counter < len(curropted_masks):                            
                car_code, angle_code = filename_to_code(curropted_masks[counter])
                # print (car_code, angle_code)
                image = read_image(car_code, angle_code)
                ax[i, j].imshow(image)

                mask = read_image(car_code, angle_code, True)
#                 mix = cv2.bitwise_and(image, image, mask = mask)                    
                ax[i, j].imshow(mask, alpha = 0.9)
                ax[i, j].set_title(curropted_masks[counter])
    #                 ax[i, j].imshow(mix, cmap = 'Greys_r', alpha = 0.6)
                counter += 1
    plt.show()
    
def vis_manufacturer_distribution():
    metadata = data.read_metadata()
    metadata.index = metadata['id']

    train_masks = data.read_train_masks()
    train_ids = train_masks['img'].apply(lambda x: x[:-7])
    train_ids = list(set(train_ids))
    train_metadata = metadata.loc[train_ids]
    plt.figure(figsize=(12, 10))
    # train_metadata = 
    sns.countplot(y="make", data=train_metadata)
    plt.show()

def plot_manufacturer_stats():
    man_history_DF = pd.read_csv('logs/man.csv')
    fig, axes = plt.subplots(nrows = 1, ncols = 2, figsize=(10, 5))
    ax = man_history_DF[['acc', 'val_acc']].plot(ax = axes[0]);
    ax.set_title('model accuracy')
    ax.set_xlabel('epoch')
    ax.set_ylabel('accuracy')
    # pd.DataFrame(baseline_history.history)[['acc', 'val_acc']].plot()
    ax = man_history_DF[['loss', 'val_loss']].plot(ax = axes[1])
    ax.set_title('model loss')
    ax.set_xlabel('epoch')
    ax.set_ylabel('loss')
    fig.show()


def plot_final_results():
    """ Plots a comparison score between the three models"""
    """ The results are hardcoded here for simplicity."""
    res = [[0.7491, 0.743401], [0.8848, 0.8894190], [0.9886, 0.989057]]
    res_pd = pd.DataFrame(res)
    res_pd.columns = ['Kaggle Score', 'Validation Score']
    ax = res_pd.plot()
    ax.set_title('Final results')
    ax.set_xlabel('Models')
    ax.set_ylabel('Score')
    ax.set_xticks(np.arange(0,3,1))
    labels=[item.get_text() for item in ax.get_xticklabels()]
    labels[0] = 'Avg Mask Benchmark'
    labels[1] = 'Simple CNN Benchmark'
    labels[2] = 'Unet (128X128)'
    ax.set_xticklabels(labels)

def plot_baseline_stats():
    baseline_history_DF = pd.read_csv('logs/baseline.csv')
    fig, axes = plt.subplots(nrows = 1, ncols = 2, figsize=(10, 5))
    # plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    baseline_history_DF[['dice_coef', 'val_dice_coef']].plot(ax = axes[0]);
    # plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    baseline_history_DF[['loss', 'val_loss']].plot(ax = axes[1])
    fig.suptitle("Learning curve for Baseline model")
    # plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    axes[0].set_xlabel('epoch')
    axes[0].set_ylabel('Dice coef')
    axes[1].set_xlabel('epoch')
    axes[1].set_ylabel('Loss')
    # plt.tight_layout()

def plot_unet128_stats():
    # baseline_history_DF = pd.DataFrame(baseline_history.history)
    unet_128_history = pd.read_csv('logs/unet_128_history.csv')
    fig, axes = plt.subplots(nrows = 1, ncols = 2, figsize=(10, 5))
    unet_128_history[['dice_coeff', 'val_dice_coeff']].plot(ax = axes[0]);
    # pd.DataFrame(baseline_history.history)[['acc', 'val_acc']].plot()
    unet_128_history[['loss', 'val_loss']].plot(ax = axes[1])
    fig.suptitle("U-Net Model")
    # plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    axes[0].set_xlabel('epoch')
    axes[0].set_ylabel('Dice coeff')
    axes[1].set_xlabel('epoch')
    axes[1].set_ylabel('Loss')
    fig.show()


def vis_predictions(model, fullRes=False):
    nrows = 3
    f, ax = plt.subplots(nrows = nrows, ncols = 3, sharex = True, sharey = True, figsize=(20,20))
    # print(ax.__class__)
    sampled_imgs = np.random.choice(train_masks['img'], nrows)
    #     sampled_imgs = [TRAIN_PATH + '/' + i for i in sampled_imgs]
    
    counter = 0

    for i in range(nrows):
        # for j in range(ncols):
        car_code, angle_code = filename_to_code(sampled_imgs[counter])
        image = read_image(car_code, angle_code)
        im = resize(image)

        x_batch = []
        x_batch.append(im)
        x_batch = np.array(x_batch, np.float32) /255
        pred = model.predict(x_batch).squeeze()

        if fullRes is True:
            im = image
            pred = cv2.resize(pred, (image.shape[1], image.shape[0]))

        ax[i, 0].imshow(image)
        ax[i, 1].imshow(pred, cmap='gray')
        ax[i, 2].imshow(pred > THRESHOLD, cmap='gray')
            
        counter += 1

    plt.show()    

def vis_predictions_ext(model, fullRes = False):
    nrows = 3
    f, ax = plt.subplots(nrows = nrows, ncols = 3, sharex = True, sharey = True, figsize=(20,20))

    samples = data.list_car_and_dog_images()

    sampled_imgs = np.random.choice(samples, nrows)
    
    for i in tqdm(range(len(sampled_imgs))):
        sample = sampled_imgs[i]
        image = ndimage.imread(sample)
        # image = cv2.imread(sample)
        try:

            im = image[:,:,:3]

            im = resize(im)
            
            x_batch = []
            x_batch.append(im)
            x_batch = np.array(x_batch, np.float32) /255
            pred = model.predict(x_batch).squeeze()

            if fullRes is True:
                im = image
                pred = cv2.resize(pred, (image.shape[1], image.shape[0]))

            filename = sample[0:-4]

            ax[i, 0].imshow(im, interpolation='nearest', aspect='auto')
            ax[i, 1].imshow(pred, cmap='gray')
            ax[i, 2].imshow(pred > THRESHOLD, cmap='gray')

        except IndexError:
            print("Issue with '{}'".format(sample))
            print(ndimage.imread(sample).shape)
        
    # plt.tight_layout()


    plt.show()    

## http://nbviewer.jupyter.org/github/raghakot/keras-vis/blob/master/examples/vggnet/activation_maximization.ipynb
def vis_activation_maximizations(model, layer_idx):
    
    from vis.visualization import visualize_activation, get_num_filters

    num_filters = get_num_filters(model.layers[layer_idx])
    print("num_filters : ", num_filters)
    fig_size = int(np.ceil(np.sqrt(num_filters)))
    print("fig_size : " , fig_size)
    layer_name = model.layers[layer_idx].name

    fig, ax = plt.subplots(fig_size, fig_size, figsize=(20, 20))
    fig.suptitle('{} {}'.format(layer_idx, layer_name))

    if(num_filters == 1):
        img = visualize_activation(model, layer_idx, filter_indices = 0)        
        # plt.imshow(img)
        ax.imshow(img)
    else:        
        for i in range(num_filters):
            img = visualize_activation(model, layer_idx, filter_indices = i)
            ax[int(i/fig_size), (i%fig_size)].imshow(img)
    
    fig.savefig('img_act/unet_128_{}_{}_act.png'.format(layer_idx, layer_name), dpi = fig.dpi)
    plt.close(fig)


def vis_all_activations(model):
    layers = model.layers
    for i, layer in enumerate(layers):
        print("Visualizing layer {}({}) activations".format(i, layer.name))
        vis_activation_maximizations(model, i)
