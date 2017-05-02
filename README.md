#** Traffic Sign Recognition** 

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

We can see that all data sets have very similar distribution of labels, so it shouldn't influence on accuracy of classification. On the other hand, it's worth to notice that some labels have less than few hundred examples so CNN may have lower accuracy on that traffic signs. I assume that this distribution reflects frequency of appearance in normal life so I left this data sets unchanged.

### Design and Test a Model Architecture

As a first step, I decided to convert the images to grayscale because it reduced my input data size three times (from 3 RGB channels to only one). I've done it by simply averaging the RGB values into one value. But I ended up with no / very few impact on accuracy so I left all images in RGB format.

Secondly, I normalized the image data as follows: new_value = (old_value - 128) / 128. 
Gradients descen converges much faster on normalized data as well as softmax function do its job much better if the data is normalized with mean 0 and the range -1 to 1.

I also converted the integer class labels into one-hot encoded labels

Here is an example of an original image:
![alt text][image4]

and normalized image:
![alt text][image3]


#### 2. Final model architecture (including model type, layers, layer sizes, connectivity, etc.) 

My final model consisted of the following layers:

| Layer         		|     Description	        					| Shape |
|:---------------------:|:---------------------------------------------:| :----:|
| Input         		| RGB image   							| 32x32x3 |
| Convolution 5x5     	| 1x1 stride, valid padding 	| 28x28x6 |
| RELU					|			Activation function			|28x28x6 |
| Max pooling	      	| 2x2 stride, 2x2 filter 			|14x14x6 	 |
| Convolution 5x5	    | 1x1 stride, valid padding  | 10x10x16 |
| RELU					|			Activation function									|10x10x16|
| Max pooling	      	| 2x2 stride, 2x2 filter				|5x5x16 |
| Fully connected		| Input = 400. Output = 120        									| 400x120|
| Dropout		| Keep_prob=0.5,  Input = 400. Output = 120    									| 400x120|
| RELU					|			Activation function									| 400x120|
| Fully connected		| Input = 120. Output = 84        									| 120x84|
| Dropout		| Keep_prob=0.5,  Input = 120. Output = 84    									|120x84|
| RELU					|				Activation function								|120x84|
| Logits				| Input = 84. Output = 43         									|84x43|

 

#### 3. Training the model

To train the model, I used an AdamOptimizer becouse it computes adaptive learning rates for each parameter. I also used regularization for the loss calculation.L2 regulatization prevents model from overfitting. Values of Learning rate and L2 hyperparam was chosen experimentally. I've trained CNN on my local computer without advanced GPU so I set bach size to 128 and number of epochs to 40. After 20s epochs train and valid accuracy didn't change a lot, having some gap between each other. Probably model is a little bit overfitting even though I've used 2 types of regularization. Next thing I could try would be add more augmentations to images such as brightness, zoom, etc., or try equalising with different approaches, such as using skimage exposure module, or doing a simple (x - mu) / (max - min)


My final model results were:
* training set accuracy of 0.983
* validation set accuracy of 0.937
* test set accuracy of 0.931

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?

The implementation shown in the [LeNet-5](http://yann.lecun.com/exdb/lenet/) seemed like a solid starting point, becouse it does good job in digits recognitions.

* What were some problems with the initial architecture?

Model was overfitting the data so I had to add some kind of regularization.

* How was the architecture adjusted and why was it adjusted? T

I've added a dropout regularization with keep_prob=0.5 in third and fourth layer as well as L2 regularization during computing loss value.

* Which parameters were tuned? 

I've plotted many times loss over epochs to see whether my learning rate is too big (curve decreases rapidly but then it descreases very slowly) or is too small (curve decreases very slowly over all epochs). The same method I used to tune up L2_param

* What are some of the important design choices and why were they chosen? 

CNN does a great job in images recognition becouse of their spatial reduction via striding property.

### Test a Model on New Images

#### 1. Here are ten German traffic signs that I found on the web:

![alt text][image2]

The images might be difficult to classify because six of them dont' exist in training data:
* Dead end (No through passing)
* End of minimum zone (End of 30 km/h minimum speed requirement)
* Falling stones (Possible falling or fallen rocks)
* Gas_station (Petrol station with Unleaded fuel)
* Max width allowed (Width limit (including wing mirrors))
* No stopping

but I wanted to find out how my model will behave in that scenario.

#### 2. Here are the results of the prediction:

| Image			        |     Prediction	        					| Status|
|:---------------------:|:---------------------------------------------:| :----:|
| Children crossing      		| Children crossing   									| OK|
| Dead end     			| Go straight or right 										|Wrong|
| End of minimum zone					| Keep right											|Wrong|
| Falling stones      		| Bumpy Road					 				|Wrong|
| Gas_station			| Right-of-way at the next intersection      							|Wrong|
| Max width allowed     		| Speed limit (70km/h)   									| Wrong|
| No stopping     			| Priority road 										|Wrong|
| Pedestrian crosswalk					| General caution									|Wrong|
| Priority road     		| Priority road				 				|OK|
| Traffic circle			| Speed limit (100km/h)      							|Wrong|


The model was able to correctly guess 2 of the 10 traffic signs, which gives an accuracy of 20%. This compares favorably to the accuracy on the test set of 93,1% seems to by very bad accuracy. But, as I mentioned above, only four out of ten traffic signs exist in training data. Two of them were predicted correctly and two of them had good prediction on the second place:

* Pedestrian crosswalk was predictied wrong, but on the second place was correct prediction with 10,12%
* Traffic circle (Roundabout mandatory) was predictied wrong, but on the second place was correct prediction with 32,19%


#### 3. Top 5 softmax probabilities for each image along with the sign type of each probability. 
![alt text][image1]



