# Machine Learning Engineer Nanodegree
## Capstone Proposal
Nitish Puri   
August 28th, 2017

## Using Deep Learning for Image Masking

### Domain Background

One of the reason I decided to do the Machine Learning Nanodegree was my growing interest in computational perception. There are several closely related perception problems that are being addressed using the rapidly emerging field of deep learning. This project is one variant of the problem of [Image Segmenation](https://en.wikipedia.org/wiki/Image_segmentation) inspired from the [Kaggle Image Masking Challenge](https://www.kaggle.com/c/carvana-image-masking-challenge).   
Historically, the problem of image segmentation has been solved by using traditional Computer vision principles like k-means clustering, thresholding, edge detection and even lossy compression techniques. These techniques require careful engineering of the image pipeline, which can differ a lot between different problems or different datasets.The problem of image segmentation has been studied in various domains ranging from robot perception to medical image analysis and the results obtained and models created are transferable to these domains as well.

The challenge is organized by [Carvana](https://www.carvana.com/). An interesting part of their innovation is a custom rotating photo studio that automatically captures and processes 16 standard images of each vehicle in their inventory. While they capture high quality photos, bright reflections and cars with similar colors as the background cause automation errors, which requires a skilled photo editor to fix. In this project we are going to develop an algorithm that automatically removes the photo studio background. This will allow Carvana to superimpose cars on a variety of backgrounds. We will be analyzing a dataset of photos, covering different vehicles with a wide variety of year, make and model combinations.
![problem](/images/carvana_graphics.png)

### Problem Statement

As mentioned in the previous section, we are provided with 16 standard images(1918 X 1280) of each vehicle in their inventory. We also have a corresponding image mask(1918 X 1280) for each input image. The problem is in automatically create an image mask for unseen images of automobiles in a similar setting. One potential solution that I could immediately come think of was using Deep Neural Network, starting with *Convolution* layers, extracting image features and then using *De-Convolution* to map the features back into original image dimensions. This would identify each pixel from the original image as either belonging to the automobile or the background. The background pixels can then be masked to create an image mask for the given image.

### Datasets and Inputs

The dataset contains a large number of car images(as .jpg files) of different makes. Each car has exactly 16 images, each one taken at different angles. Each car has a unique id and images are named according to `id_01.jpg`, `id_01.jpg`, ...   `id_16.jpg`. In addition to the images, we also have some basic metadata about the car, the make, model, year and trim.
![0cdf5b5d0ce1_04](/images/0cdf5b5d0ce1_04.jpg)
For training we are provided with with a .gif file that contains the manually cutout mask for each image.
![0cdf5b5d0ce1_04](/images/0cdf5b5d0ce1_04_mask.gif)
This dataset can be obtained from the [Kaggle Challenge](https://www.kaggle.com/c/carvana-image-masking-challenge/data).  So, the main objective of the problem is to create an image mask, given an unseen image of an automobile. However, to simplify the submissions and evaluation, the final output is required to be converted into run-length encoding on the pixel values. Instead of an exhaustive list of indices for segmentation, we will generate pair of values that contain a start position and a run length. A sample output can be found [here](sample_submission.csv).


### Solution Statement
_(approx. 1 paragraph)_

In this section, clearly describe a solution to the problem. The solution should be applicable to the project domain and appropriate for the dataset(s) or input(s) given. Additionally, describe the solution thoroughly such that it is clear that the solution is quantifiable (the solution can be expressed in mathematical or logical terms) , measurable (the solution can be measured by some metric and clearly observed), and replicable (the solution can be reproduced and occurs more than once).

### Benchmark Model
_(approximately 1-2 paragraphs)_

In this section, provide the details for a benchmark model or result that relates to the domain, problem statement, and intended solution. Ideally, the benchmark model or result contextualizes existing methods or known information in the domain and problem given, which could then be objectively compared to the solution. Describe how the benchmark model or result is measurable (can be measured by some metric and clearly observed) with thorough detail.

### Evaluation Metrics
_(approx. 1-2 paragraphs)_

In this section, propose at least one evaluation metric that can be used to quantify the performance of both the benchmark model and the solution model. The evaluation metric(s) you propose should be appropriate given the context of the data, the problem statement, and the intended solution. Describe how the evaluation metric(s) are derived and provide an example of their mathematical representations (if applicable). Complex evaluation metrics should be clearly defined and quantifiable (can be expressed in mathematical or logical terms).

### Project Design
_(approx. 1 page)_

In this final section, summarize a theoretical workflow for approaching a solution given the problem. Provide thorough discussion for what strategies you may consider employing, what analysis of the data might be required before being used, or which algorithms will be considered for your implementation. The workflow and discussion that you provide should align with the qualities of the previous sections. Additionally, you are encouraged to include small visualizations, pseudocode, or diagrams to aid in describing the project design, but it is not required. The discussion should clearly outline your intended workflow of the capstone project.

-----------

**Before submitting your proposal, ask yourself. . .**

- Does the proposal you have written follow a well-organized structure similar to that of the project template?
- Is each section (particularly **Solution Statement** and **Project Design**) written in a clear, concise and specific fashion? Are there any ambiguous terms or phrases that need clarification?
- Would the intended audience of your project be able to understand your proposal?
- Have you properly proofread your proposal to assure there are minimal grammatical and spelling mistakes?
- Are all the resources used for this project correctly cited and referenced?
