from utils.params import *

def train_manufacturer_gen():
    while True:
        img_filename = np.random.choice(train_images)        
        car_code, angle_code = filename_to_code(img_filename)
        img = read_image(car_code, angle_code)/255
        img = resize(img)
        label = pd.get_dummies(maker).loc[car_code].values
        yield img.reshape(-1, INPUT_SIZE, INPUT_SIZE, 3), label.reshape(1, -1)
    
def val_manufacturer_gen():
    while True:
        img_filename = np.random.choice(validation_images)        
        car_code, angle_code = filename_to_code(img_filename)
        img = read_image(car_code, angle_code)/255
        img = resize(img)
        label = pd.get_dummies(maker).loc[car_code].values
        yield img.reshape(-1, INPUT_SIZE, INPUT_SIZE, 3), label.reshape(1, -1)
    

