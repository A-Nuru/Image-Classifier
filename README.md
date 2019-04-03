
# **Image Classifier**


## Table of Contents
1. [Installation](https://github.com/A-Nuru/Image-Classifier#Installation)
2. [Project Motivation](https://github.com/A-Nuru/Image-Classifier#Project-Motivation)
3. [File Descriptions](https://github.com/A-Nuru/Image-Classifier#File-Descriptions)
4. [Results](https://github.com/A-Nuru/Image-Classifier#Results)
5. [Acknowledgements](https://github.com/A-Nuru/Image-Classifier#Licensing-Authors-Acknowledgements)

## Installation
The libraries employed in this project to run the code are Anaconda distribution of Python 3.*, Numpy, Pandas, MatplotLib, Pytorch, PIL and json. If you have a lower python version, you can consider an upgrade using pip or conda.

## Project Motivation
Going forward, AI algorithms will be incorporated into more and more everyday applications. For example, you might want to include an image classifier in a smartphone app. To do this, you'd use a deep learning model trained on hundreds of thousands of images as part of the overall application architecture. A large part of software development in the future will be using these types of models as common parts of applications.

In this project, an image classifier is trained to recognize different species of flowers. Imagine using something like this in a phone app that could tell the name of the flower your camera is looking at. In practice, the trained classifier will then be exported for use as an application. We'll be using this dataset of 102 flower categories.

After the project completion, we'll have an application that can be trained on any set of labelled images. Finally, the project will be learning about flowers and end up as a command line application. 

## File Descriptions
The files associated with this work include
* The jupyter ipython notebook 
* The html file generated from the jupyter ipython notebook
* The training python file and the predict python file with which this application can be run as a command line application
## Instructions
1. Train a new network on a data set with train.py

* Basic usage: python ```train.py data_directory```
* Prints out training loss, validation loss, and validation accuracy as the network trains
* Options:
Set directory to save checkpoints: python train.py data_dir --save_dir save_directory
Choose architecture: python train.py data_dir --arch "vgg13"
Set hyperparameters: python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20
Use GPU for training: python train.py data_dir --gpu
2. Predict flower name from an image with predict.py along with the probability of that name. That is, you'll pass in a single image /path/to/image and return the flower name and class probability.

* Basic usage: python predict.py /path/to/image checkpoint
* Options:
Return top KK most likely classes: python predict.py input checkpoint --top_k 3
Use a mapping of categories to real names: python predict.py input checkpoint --category_names cat_to_name.json
Use GPU for inference: python predict.py input checkpoint --gpu
The best way to get the command line input into the scripts is with the argparse module in the standard library.
## Results
An application which outputs the name of a flower image fed into it is produced. Checkout the ipynb file [here.](https://github.com/A-Nuru/Image-Classifier/blob/master/Image%20Classifier%20Project.ipynb)

## Licensing
The license of this project can be found [here](https://github.com/A-Nuru/Image-Classifier/blob/master/LICENSE)

