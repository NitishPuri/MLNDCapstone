import os

import train_val
import utils.vis as visutils

mainOptions = {
    "help" : ("Welcome to MLND Capstone : Image Automasking implementation\n"
              "[1]. Exploration : Visualize data\n"
              "[2]. Baseline 1(Using Avg Mask)\n"
              "[3]. Baseline 2(Simple 3 layer CNN)\n"
              "[4]. U-Net Model\n"
            #   "[5]. Maker Model(Experimental)\n"
              "[5]. Exit" ),
    1 : lambda : setCurrMenu(exploreOptions),
    2 : lambda : setCurrMenu(baseline1_avgMask_options),
    3 : lambda : setCurrMenu(baseline2_simpleCNN_options),
    4 : lambda : setCurrMenu(uNetOptions),
    # 5 : lambda : setCurrMenu(uNetOptions),
    5 : exit
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
              "[5].  Calculate Validation Score.\n"
              "[6].  Create submission.\n"
              "[7].  Visualize predictions on external dataset.\n"
              "[8]. Back to Main menu.\n"),

    1  : lambda : train_val.show_uNet_summary() ,
    2  : lambda : train_val.trainUnet128Model(),
    3  : lambda : visutils.plot_unet128_stats(),
    4  : lambda : train_val.vis_unet128_predictions(),
    5  : lambda : train_val.score_unet_val(),
    6  : lambda : train_val.create_unet_submission(),
    7  : lambda : train_val.vis_predictions_baseline_external(),
    8  : lambda : setCurrMenu(mainOptions)
}

baseline1_avgMask_options = {
    "help" : ("Baseline 1(Using Avg Mask)\n"
              "[1]. View Avg Mask(create if not available)\n" 
              "[2]. Calculate validation score.\n"
              "[3]. Create Submission\n"
              "[4]. Back to Main menu.\n" ),

    1  : lambda : train_val.show_avg_mask(),
    2  : lambda : train_val.score_avg_baseline_val_score(),
    3  : lambda : train_val.create_avgMask_submission(),
    4  : lambda : setCurrMenu(mainOptions)
}


baseline2_simpleCNN_options = {
    "help" : ("Baseline 2 (Using Vanilla CNN)..\n"
              "[1].  Model Summary\n" 
              "[2].  Train model\n"
              "[3].  Plot training summary\n"
              "[4].  Visualize sample predictions\n"
              "[5].  Calculate Validation score\n"
              "[6].  Create submission.\n"
              "[7].  Visualize predictions on external dataset.\n"
              "[8].  Back to Main menu.\n"),

    1  : lambda : train_val.show_baseline_2_summary(),
    2  : lambda : train_val.trainBaselineModel(),
    3  : lambda : visutils.plot_baseline_stats(),
    4  : lambda : train_val.vis_baseline_predictions() ,
    5  : lambda : train_val.score_baseline_val(),
    6  : lambda : train_val.create_baseline_submission(),
    7  : lambda : train_val.vis_predictions_baseline_external(),
    8  : lambda : setCurrMenu(mainOptions)
}

maker_model_options = {
    "help" : ("Manufacturer Model (Guess the maker, experimental)..\n"
              "[1].  Model Summary\n" 
              "[2].  Train model\n"
              "[3].  Plot training summary\n"
              "[4].  Visualize sample predictions\n"
              "[5].  Visualize layer activations.\n"
              "[6]. Back to Main menu.\n"),

    1  : lambda : train_val.show_manufacturerModel_summary(),
    2  : lambda : train_val.trainManufacturerModel(),
    3  : lambda : visutils.plot_manufacturer_stats(),
    5  : lambda : visutils.vis_dataset(nrows = 2, ncols = 2, mask_alpha = 0.0, augment = True),
    # 6  : lambda : visutils.vis_manufacturer_distribution(),
    6  : lambda : setCurrMenu(mainOptions)
}


# Initialize Current menu with Main menu options.
currMenu = mainOptions

def setCurrMenu(menu):
    global currMenu
    currMenu = menu

if __name__=="__main__":
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
