


def explore():
    print ("Exploring data..")



options = {
    1 : explore
}



if __name__=="__main__":
    print("Welcome to MLND Capstone : Image Automasking implementation")
    print("1. Exploration : Visualize data")
    print("1. Train U-Net 128 model")
    print("2. Predict using U-Net 128 model")
    print("4. Exploration : Visualize data")

    choice = int(input())

    print("You choose :", choice)

    options[choice]()

    # trainManufacturerModel()