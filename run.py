import os

import cv2
import matplotlib.pyplot as plt
from keras.utils import plot_model

import utils.vis as visutils
import train_val
import utils.zf_baseline as zf_baseline

mainOptions = {
    "help" : ("Welcome to MLND Capstone : Image Automasking implementation\n"
              "[1]. Exploration : Visualize data\n"
              "[2]. U-Net Model Analysis\n"
              "[3]. Baseline 1(Using Avg Mask)\n"
              "[4]. Baseline 2(Simple 3 layer CNN)\n"
              "[5]. Maker Model(Experimental)\n"
              "[6]. Exit" ),
    1 : lambda : setCurrMenu(exploreOptions),
    2 : lambda : setCurrMenu(uNetOptions),
    3 : lambda : setCurrMenu(baseline1_avgMask_options),
    4 : lambda : setCurrMenu(baseline2_simpleCNN_options),
    5 : lambda : setCurrMenu(uNetOptions),
    6 : exit
}

exploreOptions = {
    "help" : ("Exploring data..\n"
              "[1]. Explore raw data(without input masks)\n" 
              "[2]. Explore raw data(with input masks)\n"
              "[3]. Explore examples of incorrectly labeled data(with input masks)\n"
              "[4]. Explore examples of augmented image samples(with input masks)\n"
              "[5]. Explore examples of augmented image samples(without input masks)\n"
              "[6]. Visualize Car maker distribution in the dataset.\n"
              "[7]. Back to Main Menu\n"),

    1 : lambda : visutils.vis_dataset(nrows = 3, ncols = 3, mask_alpha = 0.0, augment = False),
    2 : lambda : visutils.vis_dataset(nrows = 3, ncols = 3, mask_alpha = 0.4, augment = False),
    3 : lambda : visutils.vis_curropted_dataset(),
    4 : lambda : visutils.vis_dataset(nrows = 2, ncols = 2, mask_alpha = 0.4, augment = True),
    5 : lambda : visutils.vis_dataset(nrows = 2, ncols = 2, mask_alpha = 0.0, augment = True),
    6 : lambda : visutils.vis_manufacturer_distribution(),
    7 : lambda : setCurrMenu(mainOptions)
}

uNetOptions = {
    "help" : ("U-Net Model..\n"
              "[1].  Model Summary\n" 
              "[2].  Train model\n"
              "[3].  Plot training summary\n"
              "[4].  Visualize sample predictions\n"
              "[5].  Create submission.\n"
              "[6].  Visualize layer activations.\n"
              "[7].  Visualize filters(using activation maximization).\n"
              "[8].  Visualize predictions on external images(car, experimental).\n"
              "[9]. Visualize predictions on external images(notCar, experimental).\n"
              "[10]. Back to Main menu.\n"),

    1  : lambda : train_val.show_uNet_summary() ,
    2  : lambda : train_val.trainUnet128Model(),
    # 3  : lambda : visutils.vis_dataset(nrows = 2, ncols = 2, mask_alpha = 0.4, augment = True),
    # 4  : lambda : visutils.vis_dataset(nrows = 2, ncols = 2, mask_alpha = 0.0, augment = True),
    # 5  : lambda : visutils.vis_manufacturer_distribution(),
    # 6  : lambda : setCurrMenu(mainOptions),
    # 7  : lambda : setCurrMenu(mainOptions),
    # 8  : lambda : setCurrMenu(mainOptions),
    # 9  : lambda : setCurrMenu(mainOptions),
    10 : lambda : setCurrMenu(mainOptions)
}

baseline1_avgMask_options = {
    "help" : ("Baseline 1(Using Avg Mask)\n"
              "[1]. View Avg Mask(create if not available)\n" 
              "[2]. Calculate validation score.\n"
              "[3]. Create Submission\n"
              "[4]. Back to Main menu.\n" ),

    1  : lambda : train_val.show_avg_mask(),
    2  : lambda : train_val.score_baseline_val_score(),
    3  : lambda : train_val.create_avgMask_submission(),
    4  : lambda : setCurrMenu(mainOptions)
}


baseline2_simpleCNN_options = {
    "help" : ("Baseline 2 (Using Vanilla CNN)..\n"
              "[1].  Model Summary\n" 
              "[2].  Train model\n"
              "[3].  Plot training summary\n"
              "[4].  Visualize sample predictions\n"
              "[5].  Create submission.\n"
              "[6].  Visualize layer activations.\n"
              "[7].  Visualize filters(using activation maximization).\n"
              "[8].  Visualize predictions on external images(car, experimental).\n"
              "[9].  Visualize predictions on external images(notCar, experimental).\n"
              "[10]. Back to Main menu.\n"),

    1  : lambda : train_val.show_baseline_2_summary(),
    2  : lambda : train_val.trainBaselineModel(),
    # 3  : lambda : visutils.vis_dataset(nrows = 2, ncols = 2, mask_alpha = 0.4, augment = True),
    # 4  : lambda : visutils.vis_dataset(nrows = 2, ncols = 2, mask_alpha = 0.0, augment = True),
    # 5  : lambda : visutils.vis_manufacturer_distribution(),
    # 6  : lambda : setCurrMenu(mainOptions),
    # 7  : lambda : setCurrMenu(mainOptions),
    # 8  : lambda : setCurrMenu(mainOptions),
    # 9  : lambda : setCurrMenu(mainOptions),
    10 : lambda : setCurrMenu(mainOptions)
}

maker_model_options = {
    "help" : ("Manufacturer Model (Guess the maker, experimental)..\n"
              "[1].  Model Summary\n" 
              "[3].  Train model\n"
              "[4].  Plot training summary\n"
              "[5].  Visualize sample predictions\n"
              "[6].  Visualize layer activations.\n"
              "[7]. Back to Main menu.\n"),

    1  : lambda : train_val.show_manufacturerModel_summary(),
    3  : lambda : train_val.trainManufacturerModel(),
    # 4  : lambda : visutils.vis_dataset(nrows = 2, ncols = 2, mask_alpha = 0.4, augment = True),
    # 5  : lambda : visutils.vis_dataset(nrows = 2, ncols = 2, mask_alpha = 0.0, augment = True),
    # 6  : lambda : visutils.vis_manufacturer_distribution(),
    7  : lambda : setCurrMenu(mainOptions)
}


# Initialize Current menu with Main menu options.
currMenu = mainOptions

def setCurrMenu(menu):
    global currMenu
    currMenu = menu

if __name__=="__main__":
    # os.system('cls')
    while(True):
        os.system('cls')
        print(currMenu["help"]) 
        try:
            choice = int(input())
            currMenu[choice]()
        except ValueError:
            print("ValueError,..")
            print("Unimplemented Option,..")
            input("Press Enter to continue...")
        except KeyError:
            print("KeyError,..")
            print("Unimplemented Option,..")
            input("Press Enter to continue...")
            pass