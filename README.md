# Machine Learning Engineer Nanodegree
## Specializations
## Project: Capstone Project : Image Segmentation using Deep Neural Networks


This Capstone is based on the [Carvana Image Masking](https://www.kaggle.com/c/carvana-image-masking-challenge) challenge from Kaggle. 

The Capstone project is divided into two stages, [proposal](capstone_proposal.md) and [implementation](capstone_report.md). 

### Steps to run
* The implementation uses Python 3.5. For Neural Network implementation I use Keras 2.0.
* Make sure you have all the libraries in place.
```
pip install -r requirements.txt
```
* Download the data from [here](https://www.kaggle.com/c/carvana-image-masking-challenge/data).
    You need to place and extract the downloaded data into `/input` directory. See [here](./input/Readme.md) for the final directory structure.
* Once the data is in place, you can launch the command line interface,
```
python run.py
```
* This script would present you with options to *explore* the dataset, *train and validate* baseline and unet models, *visualize predictions* and *create submissions* in the form required by kaggle.
```
Welcome to MLND Capstone : Image Automasking implementation
[1]. Exploration : Visualize data
[2]. Baseline 1(Using Avg Mask)
[3]. Baseline 2(Simple 3 layer CNN)
[4]. U-Net Model
[5]. Exit

Enter your choice...
```
* The script is divided up into similar sections and you would get options in the same form.
* For more documentation about the project you can refer [run.py](run.py) and [capstone_report.md](capstone_report.md).

----------------------------------------------------------------------------

Please email [npuri1903@gmail.com](mailto:npuri1903@gmail.com) if you have any questions.
