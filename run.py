import os
import utils.vis as visutils


exploreOptions = {
    "help" : ("Exploring data..\n"
              "1. Explore raw data(without input masks)\n" 
              "2. Explore raw data(with input masks)\n"
              "3. Explore examples of incorrectly labeled data(with input masks)\n"
              "4. Explore examples of augmented image samples(with input masks)\n"
              "5. Explore examples of augmented image samples(without input masks)\n"
              "6. Visualize Car maker distribution in the dataset.\n"
              "7. Back to Main Menu\n"),

    1 : lambda : visutils.vis_dataset(nrows = 3, ncols = 3, mask_alpha = 0.0, augment = False),
    2 : lambda : visutils.vis_dataset(nrows = 3, ncols = 3, mask_alpha = 0.4, augment = False),
    3 : lambda : visutils.vis_curropted_dataset(),
    4 : lambda : visutils.vis_dataset(nrows = 2, ncols = 2, mask_alpha = 0.4, augment = True),
    5 : lambda : visutils.vis_dataset(nrows = 2, ncols = 2, mask_alpha = 0.0, augment = True),
    6 : lambda : visutils.vis_manufacturer_distribution(),
    7 : lambda : setCurrMenu(mainOptions)
}


# def unet_analysis:
#     pass

# def baseline_avg_mask:
#     pass

# def baseline_simple_cnn:
#     pass

# def maker_model_analysis:
#     pass    

mainOptions = {
    "help" : ("Welcome to MLND Capstone : Image Automasking implementation\n"
              "1. Exploration : Visualize data\n"
              "2. U-Net 128\n"
              "3. Baseline 1(Using Avg Mask)\n"
              "4. Baseline 2(Simple 3 layer CNN)\n"
              "5. Maker Model(Experimental)\n"
              "6. Exit" ),
    1 : lambda : setCurrMenu(exploreOptions),
    # 2 : unet_analysis,
    # 3 : baseline_avg_mask,
    # 4 : baseline_simple_cnn,
    # 5 : maker_model_analysis,
    6 : exit
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
        choice = int(input())
        currMenu[choice]()
