# **Behavioral Cloning**

## Writeup Report

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* models.py containing the script to create the model.
* train.py for training the model.
* image_generator.py containing the script to generate and augment the training and validation set.
* drive.py for driving the car in autonomous mode
* ./train_models/model.h5 containing a trained convolution neural network
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py ./train_models/model.h5
```

#### 3. Submission code is usable and readable

The models.py file contains the code for creating the model class. Also, the file has two classes. The first one is a model which was performed for transfer learning by using VGG16 architecture. The second one is a model which is published by the autonomous vehicle at Nvidia.<br>
The class has training, saving and loading functions.

The train.py file contains the code for training the model which was loaded or created from models.py.
The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consisted of the following layers(models.py lines 81 - 99):

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 160x320x3 RGB image   							|
| Cropping     	| cropping 50 rows pixels from the top of the image, crop 20 rows pixels from the bottom of the image 	|
| Resize | Resize 66x200x3  |
| Normalization | Normalize to -0.5 ~ 0.5. |
| Convolution 5x5  | 2x2 stride, valid, out:31x98x24 |
| Convolution 5x5  | 2x2 stride, valid, out:14x47x36 |
| Convolution 5x5  | 2x2 stride, valid, out:5x22x48  |
| Convolution 3x3  | 1x1 stride, valid, out:3x20x64  |
| Convolution 3x3  | 1x1 stride, valid, out:1x18x64  |
| Fully connected  | activation: RELU, out:100       |
| Dropout          | rate:0.5                        |
| Fully connected  | activation: RELU, out:100       |
| Dropout          | rate:0.5                        |
| Fully connected  | activation: RELU, out:10        |
| Prediction angle | no activation, out:1            |



#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (models.py lines 95, 97).

The model was trained and validated on different data sets to ensure that the model was not overfitting . The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (models.py line 103).

#### 4. Appropriate training data

For the learning data, I used the image running in the middle of the lane. I trained the model using the image of the left and right of the car, the center.
Also, I created data to go back to the center of the lane when the model went off the track or went to the left and right sides of the lane.

For details about how I created the training data, see the next section.

### Archinecture and Training Documentation

#### 1. Solution Design Approach

The overall strategy is to create a model that takes the image as input and to output steering angle.

My first step was to transfer learning using the VGG16 architecture. I kept the first two convolution layers and inserted three fully-connected layers on the output side. I trained the model. I tried running the trained model with a simulator, but it did not work. looking at the mean-squared-error, both the training data and the validation data had large values. In other words, it was underfitting. I thought that pre-trained models can not extract features for driving end-to-end.

Next, I decided to make the model architecture which was released by the autonomous vehicles team at the Nvidia. This document is [Here](https://devblogs.nvidia.com/deep-learning-self-driving-cars/).
At first, training was done using only the image of the center camera. However, when I ran it with the simulator, it went off the track. training was not done well. I thought that it was not possible to recognize the lane as a feature with only the center camera. So I trained the model using the images of the center, left and right cameras. I tried running it with a simulator and ran well in the center of the lane.
The mean squared error of this model is shown below.
<img src='./writeup_iamges/mean_squared_error.png'>


#### 2. Architecture Documentation


Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
