# **Traffic Sign Recognition**

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image_distribution]: ./images/training-data-distribution.png "Training Data Distribution"
[image_original]: ./images/original.png "Original"
[image_grayscale]: ./images/grayscale.png "Grayscale"
[image_blur]: ./images/blur.png "Blur"
[image_rotate]: ./images/rotate.png "Rotate"
[image_darken]: ./images/darken.png "Darken"
[image_brighten]: ./images/brighten.png "Brighten"
[image_new_1]: ./new-signs/new-sign-1.jpg "Traffic Sign 1"
[image_new_2]: ./new-signs/new-sign-2.jpg "Traffic Sign 2"
[image_new_3]: ./new-signs/new-sign-3.jpg "Traffic Sign 3"
[image_new_4]: ./new-signs/new-sign-4.jpg "Traffic Sign 4"
[image_new_5]: ./new-signs/new-sign-5.jpg "Traffic Sign 5"

---

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is *34799*
* The size of the validation set is *4410*
* The size of test set is *12630*
* The shape of a traffic sign image is *(32, 32, 3)*
* The number of unique classes/labels in the data set is *43*

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the training data for each class is distributed.

![alt text][image_distribution]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because I'm using LeNet-5 as a starting point.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image_original]
![alt text][image_grayscale]

As a last step, I normalized the image data from 0 <-> 255 to 0 <-> 1 because I want to reduce the variance in the data.

I decided to generate additional data because some classes have very few samples.

To add more data to the the data set, I used the following techniques:

##### Blur

Here is an example of an original image and an augmented image:

![alt text][image_original]
![alt text][image_blur]

##### Rotate

Here is an example of an original image and an augmented image:

![alt text][image_original]
![alt text][image_rotate]

##### Darken

Here is an example of an original image and an augmented image:

![alt text][image_original]
![alt text][image_darken]

##### Brighten

Here is an example of an original image and an augmented image:

![alt text][image_original]
![alt text][image_brighten]


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x1 Grayscale image   							|
| Convolution     	| 1x1 stride, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution     	| 1x1 stride, outputs 10x10x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				|
| Flatten	      	| outputs 400 				|
| Fully connected		| outputs 120        									|
| RELU					|												|
| Fully connected		| outputs 84        									|
| RELU					|												|
| Fully connected		| outputs 43        									|


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used the following settings:

* Batch size: 128
* Learning rate: 0.001
* Number of epochs: 20

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of ?
* validation set accuracy of ?
* test set accuracy of ?

I have chosen the LeNet-5 architecture because it could achieve 0.93 or more accuracy with the augmented training data.


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image_new_1] ![alt text][image_new_2] ![alt text][image_new_3]
![alt text][image_new_4] ![alt text][image_new_5]

Traffic sign 1, 4, and 5 might be difficult to classify because they are viewed from an angle instead of from the front.
Traffic sign 3 and 4 might be difficult to classify because there are watermarks on them.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image | Prediction |
|:---------------------:|:---------------------------------------------:|
| Right-of-way at the next intersection | Right-of-way at the next intersection |
| Yield | Yield |
| 30 km/h | 30 km/h |
| Pedestrians | Right-of-way at the next intersection |
| Stop Sign | Stop sign |


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

For image 1, 2, 3 and 4, the model is very sure about its prediction (probability of 1.0, although image 4 is incorrectly classified). For image 5, the model is less sure (probability of 0.99) and is also misclassified.

##### Sign 1:

| Prediction | Probability |
|:---------------------:|:---------------------------------------------:|
| Right-of-way at the next intersection | 1.0 |
| Double curve | 1.64935908583944e-27 |
| Beware of ice/snow | 1.906662269546114e-32 |
| Speed limit (20km/h) | 0.0 |
| Speed limit (30km/h) | 0.0 |

##### Sign 2:

| Prediction | Probability |
|:---------------------:|:---------------------------------------------:|
| Yield | 1.0 |
| Speed limit (20km/h) | 0.0 |
| Speed limit (30km/h) | 0.0 |
| Speed limit (50km/h) | 0.0 |
| Speed limit (60km/h) | 0.0 |

##### Sign 3:

| Prediction | Probability |
|:---------------------:|:---------------------------------------------:|
| Speed limit (60km/h) | 1.0 |
| Speed limit (50km/h) | 2.4345279833570616e-27 |
| Speed limit (30km/h) | 1.3638542114285587e-29 |
| Speed limit (80km/h) | 2.516955801266155e-37 |
| Speed limit (20km/h) | 0.0 |

##### Sign 4:

| Prediction | Probability |
|:---------------------:|:---------------------------------------------:|
| General caution | 1.0 |
| Children crossing | 2.4976792856179486e-10 |
| Pedestrians | 5.833848951334496e-20 |
| Go straight or right | 4.2743047227537744e-30 |
| Right-of-way at the next intersection | 8.046355809662501e-32 |

##### Sign 5:

| Prediction | Probability |
|:---------------------:|:---------------------------------------------:|
| Speed limit (60km/h) | 0.996278703212738 |
| Stop | 0.0037213056348264217 |
| Speed limit (80km/h) | 1.346445849779998e-12 |
| Yield | 3.6636286452479405e-13 |
| Turn left ahead | 2.418657993070311e-13 |
