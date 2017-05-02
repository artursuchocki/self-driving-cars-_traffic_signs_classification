#**Traffic Sign Recognition** 

In this project, I will use deep neural networks and convolutional neural networks to classify traffic signs. Specifically, I'll train a model to classify traffic signs from the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset).

Download the [dataset](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/5898cd6f_traffic-signs-data/traffic-signs-data.zip). This is a pickled dataset in which there are already resized the images to 32x32.

> **Note**: Here is a [link](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/writeup_template.md) to README.md file describing in details my implementation and results.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./writeup_images/examples_5_softmax.jpg
[image2]: ./writeup_images/examples_german_signs.jpg
[image3]: ./writeup_images/examples_X_train_normalized.jpg
[image4]: ./writeup_images/examples_X_train.jpg
[image5]: ./writeup_images/hist_y_test.jpg
[image6]: ./writeup_images/hist_y_train.jpg
[image7]: ./writeup_images/hist_y_valid.jpg
[image8]: ./writeup_images/predicted_german_signs.jpg

---
### README

You're reading it! and here is a link to my [project code](https://github.com/artursuchocki/self-driving-cars-_traffic_signs_classification/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. 

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

Here is an exploratory visualization of the data set. It is a bar chart showing how the data is distributed across the labels in train, validation and test sets.

![alt text][image6]
![alt text][image7]
![alt text][image5]

We can see that all data sets have very similar distribution of labels, so it shouldn't influence on accuracy of classifacation. On the other hand, it's worth to notice that some labels have less than few hundred examples so CNN may have lower accuracy on that traffic signs. I assume that this distribution reflects frequency of appearance in normal life so I leave this data sets unchanged.

### Design and Test a Model Architecture

As a first step, I decided to convert the images to grayscale because it reduced my input data size three times (from 3 RGB channels to only one). I've done it by simply averaging the RGB values into one value. But I ended up with no / very few impact on accuracy so I left all images in RGB format.

As a last step, I normalized the image data as follows: new_value = (old_value - 128) / 128. Gradients descen converges much faster on normalized data as well as softmax function do its job much better if the data is normalized with mean 0 and the range -1 to 1.

I also converted the integer class labels into one-hot encoded labels

Here is an example of an original image:
![alt text][image4]
and an normalized image:
![alt text][image3]


#### 2. Describing what my final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) 

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16       									|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				|
| Fully connected		| Input = 400. Output = 120        									|
| Dropout		| Keep_prob=0.5,  Input = 400. Output = 120    									|
| RELU					|												|
| Fully connected		| Input = 120. Output = 84        									|
| Dropout		| Keep_prob=0.5,  Input = 120. Output = 84    									|
| RELU					|												|
| Logits				| Input = 84. Output = 43         									|

 

#### 3. Describing how I trained your model. The discussion include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an AdamOptimizer becouse it computes adaptive learning rates for each parameter. I also used regularization for the loss calculation.L2 regulatization prevents model from overfitting. Values of Learning rate and L2 hyperparam was chosen experimentally. I've traineg CNN on my local computer without advanced GPU so I set bach size to 128 and number of epochs to 40. After 20s epochs train and valid accuracy didn't change a lot, having some gap between each other. Probably model is a little bit overfitting even though I've used 2 types of regularization. Next thing I could try would be add more augmentations to images such as brightness, zoom, etc., or try equalising with different approaches. Such as using skimage exposure module, or doing a simple (x - mu) / (max - min)


My final model results were:
* training set accuracy of 0.983
* validation set accuracy of 0.937
* test set accuracy of 0.931

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
The implementation shown in the [LeNet-5](http://yann.lecun.com/exdb/lenet/) seemed like a solid starting point, becouse it does good job in digits recognitions.

* What were some problems with the initial architecture?
Model was overfitting the data so I had to add some kind of regularization.

* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
I've added a dropout regularization with keep_prob=0.5 in third and fourth layer as well as L2 regularization during computing loss value.

* Which parameters were tuned? How were they adjusted and why?
I've plotted many times loss over epochs to see whether my learning rate is too big (curve decreases rapidly but then it descreases very slowly) or is too small (curve decreases very slowly over all epochs). The same method I used to tune up L2_param

* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? 
CNN does a great job in images recognition becouse of their spatial reduction via striding property.

### Test a Model on New Images

#### 1. Choose ten German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are ten German traffic signs that I found on the web:

![alt text][image2]

The first image might be difficult to classify because ...

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Children crossing      		| Children crossing   									| 
| Dead end     			| Go straight or right 										|
| End of minimum zone					| Keep right											|
| Falling stones      		| Bumpy Road					 				|
| Gas_station			| Right-of-way at the next intersection      							|

| Max width allowed     		| Speed limit (70km/h)   									| 
| No stopping     			| Priority road 										|
| Pedestrian crosswalk					| General caution									|
| Priority road     		| Priority road				 				|
| Traffic circle			| Speed limit (100km/h)      							|


The model was able to correctly guess 2 of the 10 traffic signs, which gives an accuracy of 20%. This compares favorably to the accuracy on the test set of 93,1% seems to by very bad accuracy. Atfer checking the train data set it seems like there aren't traffic signs like:
* Dead end (No through passing)
* End of minimum zone (End of 30 km/h minimum speed requirement)
* Falling stones (Possible falling or fallen rocks)
* Gas_station (Petrol station with Unleaded fuel)
* Max width allowed (Width limit (including wing mirrors))
* No stopping

* Pedestrian crosswalk was predictied wrong, but on the second place was correct prediction with 10,12%
* Traffic circle (Roundabout mandatory) was predictied wrong, but on the second place was correct prediction with 32,19%


#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. 

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were:
![alt text][image1]



