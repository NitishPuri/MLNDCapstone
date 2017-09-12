#### Input Folder

Here the input data for training and testing will be placed.

Download the data from [here](https://www.kaggle.com/c/carvana-image-masking-challenge/data).   

Apart from this we also test this model on samples external to the dataset. These samples are already present in the directory `car_and_dog`.    


After extracting the data, your directory structure should look something like this.   
```
input
+--- metadata.csv
+--- Readme.md
+--- sample_submission.csv     
+--- train_masks.csv
+--- car_and_dog        # Already in the repo
|    +--- *.jpg 
+--- test
|    +--- *.jpg
+--- train 
|    +--- *.jpg
+--- train_masks
|    +--- *.gif 
```