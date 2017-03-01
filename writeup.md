#**Traffic Sign Recognition** 

## Train a Neural Network model to classify traffic signs from the German Traffic Sign Dataset
### Udacity SELF-DRIVING CAR Nanodegree Term1 Project2
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

[image1]: ./examples/visualization.png "Visualization"
[image2]: ./examples/grayscale.png     "Grayscaling"
[image3]: ./examples/Normalizing.png   "Normalizing"
[image4]: ./examples/ahead.png      "Traffic Sign 1"
[image5]: ./examples/nopassing.png  "Traffic Sign 2"
[image6]: ./examples/pedestrain.png "Traffic Sign 3"
[image7]: ./examples/priority.png   "Traffic Sign 4"
[image8]: ./examples/roundabout.png "Traffic Sign 5"
[image9]: ./examples/prediction.png "New Images Prediction"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/neo-cc/CarND-Traffic-Sign-Classifier-Project)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in the 1st and 2nd code cell of the IPython notebook.  

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 39209
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

The traffic-signs-data data combines with sepereate train validation data. I combined them together for further pre-processing. Will split them after pre-processing. 


####2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in the 3rd code cell of the IPython notebook.  

Here is an exploratory visualization of the data set. It is a bar chart showing how the data histogram for different class signs. We can see some of the classes have much larger data quntities compred to others. I believe the model may be biased towards predicting an unknown sign to a class which has more data. Better solution would be data augmentation to make the data evenly distributed. 

![alt text][image1]

I also print out all the signames in 4th code cell to get familiar with what kind of classes are there in the data. 
Then I plot all classes with 1 image in 5th code cell to get familiar the real traffic sign images.

###Design and Test a Model Architecture

####1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

The code for this step is contained in the 6-8th code cell of the IPython notebook.

As a first step, I shuffled the training data because they were in ascending order. I don't want to distort training by only providing 1 kind of class in a batch. 

I also normalized the image data because normalizing the data gives me higher validation and training accuracy after serveal tries. This process can make training faster and reduce the chances of getting stuck in local optima.

![alt text][image3]

As a last step, I decided to convert the images to grayscale because I think traffic sign classifier model depends more on the shape or content of the image which instead of color information. Grayscaling helps to reduce input layer from 3 to 1, which results in much less computing work during training.

![alt text][image2]


####2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

The code for splitting the data into training and validation sets is contained in the 9th code cell of the IPython notebook.  

To cross validate my model, I randomly split the training data into a training set and validation set. I did this by using sklearn.model_selection train_test_split function so the model would not cheat during the training process.

My final training set had 39209*0.8=31367 number of images. My validation set and test set had 39209*0.2=7842 and 12630 number of images.

I did not augment the data set. If given more time I would consider doing it because it would increase the test accuracy for sure. To add more data to the the data set, I can use the techniques such as rotations, translations and shearing. Karas has tensorflow data augmentation which could be done easily. 

##### [reference] https://medium.com/@vivek.yadav/dealing-with-unbalanced-data-generating-additional-data-by-jittering-the-original-image-7497fe2119c3#.v1haexamf


####3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the 11th cell of the ipython notebook. 

My final model used LeNet and consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image   					| 
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 28x28x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x16 				|
| Convolution 3x3	    | 1x1 stride, valid padding, outputs 10x10x16	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 					|
| Fully connected		| outputs 120 flat 								|
| RELU					|												|
| Dropout 				| 0.8   										|
| Fully connected		| outputs 84 flat 								|
| RELU					|												|
| Dropout 				| 0.8   										|
| Fully connected		| outputs 43 nclasses 							|


####4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the 13-14th cell of the ipython notebook. 

To train the model, I used the same training methed as provided in LeNet project. I didn't optimized the following parameters too much. 

Parameters:
Learning rate (initial):  0.001
Training epochs:  150
Batch size:  100
Dropout (fc):  0.8
Padding:  VALID
weights_mean:  0.0
weights_stddev:  0.1
biases_mean:  0.0


####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the 15th cell of the Ipython notebook.

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?

I used LeNet architecture. LeNet shows a very good test accuarcy result on images of numbers. It should be able to classifer with more classes on traffic signs so I tried with it at the beginning. And the result looks promising. 

Actually at first I forgot to change the n_classes to 43. It was using some hard-coded value as 10 so no matter how I changed the architecture or optimized the parameters it only got 5-6% accuracy, which made me so frustrated. Later I checked all the code then correct this bug. Finally the accuracy went up to more than 95%.

The final validation accuracy is around 0.98 after 30 epochs. I set it to stop train when validation accuracy is above 0.98 because of the training time limitation (too long). I should have trained it till the end of target epoches. 

My final model results were:
* training set accuracy of 0.991
* validation set accuracy of 0.982
* test set accuracy of 0.915

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

I manually modified the Image to 32x32 size. The resolution after the modification on size looks terriable, which could result in a bad prediction on these images.
Then I used the same pre-processing method for all the 5 images before the test.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the tenth cell of the Ipython notebook.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Ahead Only 			| Ahead Only    								|
| No Passing     		| No entry   					 				|
| Pedestrians      		| Turn left ahead   							| 
| Priority road			| Priority road      							|
| Roundabout mandatory	| Roundabout  mandatory    						|

![alt text][image9]

The model was able to correctly guess only 3 of the 5 traffic signs, which gives an accuracy of 60%. This compares negatively to the accuracy on the test set which has more than 90% accuracy. 

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a Ahead only (probability of 1.00), and the image does contain Ahead only. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         			| Stop sign   									| 
| .00     				| Speed limit (60km/h)							|
| .00					| Road work  									|
| .00	      			| Turn left ahead   			 				|
| .00				    | Yield              							|


For the second image, the model predicts that this is a No entry (probability of 1.00). But it's actually No passing. Compared these 2 images they have similar shape, only the color is sligtly different. And No entry has much more train data set than No passing, which confirmed my thoughts that model would bias with the data it has been trained more. It might help if I don't do the grayscaling in this case. The top five soft max probabilities were


| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .99         			| No entry   									| 
| .01     				| No passing         							|
| .00					| Speed limit (120km/h)  						|
| .00	      			| Roundabout mandatory  		 				|
| .00				    | Yield             							|

For the third image, the model is relatively sure that this is a Turn left ahead (probability of 1.00), however the image is Pedestrians. When I checked this input image, it does not match the Pedestrians in the traffic sign class, which means it is a completely new class. So the model tried to find a most similar sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
|1.00         			| Turn left ahead 								| 
| .00     				| Speed limit (60km/h)  						|
| .00					| End of all speed and passing limits   		|
| .00	      			| No passing					 				|
| .00				    | No passing for vehicles over 3.5 metric tons	|


For the fourth image, the model is relatively sure that this is a Priority road (probability of 1.00). The top five soft max probabilities were

| Probability         	|     Prediction	        			    		| 
|:---------------------:|:-------------------------------------------------:| 
| 1.00         			| Priority road 					     			| 
| .00     				| End of no passing by vehicles over 3.5 metric tons|
| .00					| Roundabout mandatory					    		|
| .00	      			| General caution				 		    		|
| .00				    | Right-of-way at the next intersection	    		|


For the fifth image, the model is not so sure that this is a Roundabout mandatory (probability of only 0.385). The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .38         			| Roundabout mandatory							| 
| .32     				| Vehicles over 3.5 metric tons prohibited		|
| .20					| Priority road          						|
| .08	      			| End of no passing  			 				|
| .06				    | End of speed limit (80km/h)					|


