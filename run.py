import utils.vis as visutils


def explore():
    print("Exploring data..")
    print("1. Explore raw data(without input masks)")
    print("2. Explore raw data(with input masks)")
    print("3. Explore examples of incorrectly labeled data(with input masks)")
    print("4. Explore examples of augmented image samples(with input masks)")
    print("5. Explore examples of augmented image samples(without input masks)")
    print("6. Visualize Car maker distribution in the dataset.")
    choice = int(input())

    exploreOptions[choice]()



exploreOptions = {
    1 : lambda : visutils.vis_dataset(nrows = 3, ncols = 3, mask_alpha = 0.0, augment = False),
    2 : lambda : visutils.vis_dataset(nrows = 3, ncols = 3, mask_alpha = 0.4, augment = False),
    3 : lambda : visutils.vis_curropted_dataset(),
    4 : lambda : visutils.vis_dataset(nrows = 2, ncols = 2, mask_alpha = 0.4, augment = True),
    5 : lambda : visutils.vis_dataset(nrows = 2, ncols = 2, mask_alpha = 0.0, augment = True),
    6 : lambda : visutils.vis_manufacturer_distribution()
}

mainOptions = {
    1 : explore,
    5 : exit
}

if __name__=="__main__":
    while(True):
        print("Welcome to MLND Capstone : Image Automasking implementation")
        print("1. Exploration : Visualize data")
        print("1. Train U-Net 128 model")
        print("2. Predict using U-Net 128 model")
        print("4. Exploration : Visualize data")
        print("5. Exit")

        choice = int(input())

        print("You choose :", choice)

        mainOptions[choice]()

        # trainManufacturerModel()