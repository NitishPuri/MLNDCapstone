from utils.filename import *
from utils.image import *

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

def train_generator():
    while True:
        for start in range(0, len(train_images), BATCH_SIZE):
            x_batch = []
            y_batch = []
            end = min(start + BATCH_SIZE, len(train_images))
            train_batch = train_images[start:end]                 
            for sample in train_batch :
                car_code, angle_code = filename_to_code(sample)
                img = read_image(car_code, angle_code)
                img = resize(img)
                mask = read_image(car_code, angle_code, mask = True)
                mask = resize(mask)
                img = randomHueSaturationVariation(img, hue_shift_limit=(-50,50),
                                                   sat_shift_limit=(-5, 5), val_shift_limit=(-15, 15))
                img, mask = randomShiftScaleRotate(img, mask,  rotate_limit=(-5, 5))
                img, mask = randomHorizontalFlip(img, mask)
                # image, mask = randomCrop(image, mask)

                mask = np.expand_dims(mask, axis = 2)
                
                x_batch.append(img)
                y_batch.append(mask)
                
            x_batch = np.array(x_batch, np.float32) / 255
            y_batch = np.array(y_batch, np.float32) / 255
            
            yield x_batch, y_batch
                 
def valid_generator():
    while True:
        for start in range(0, len(validation_images), BATCH_SIZE):
            x_batch = []
            y_batch = []
            end = min(start + BATCH_SIZE, len(validation_images))
            valid_batch = validation_images[start:end]
            for sample in valid_batch:
                car_code, angle_code = filename_to_code(sample)
                img = read_image(car_code, angle_code)
                img = resize(img)
                mask = read_image(car_code, angle_code, mask = True)
                mask = resize(mask)
                mask = np.expand_dims(mask, axis = 2)
                x_batch.append(img)
                y_batch.append(mask)
                
            x_batch = np.array(x_batch, np.float32) / 255
            y_batch = np.array(y_batch, np.float32) / 255
            
            yield x_batch, y_batch