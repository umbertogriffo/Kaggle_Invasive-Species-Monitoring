# Kaggle-Invasive Species Monitoring
My solutions that uses Keras (81st place LB 0.99046)

## Environment
	* 2 Intel Xeon E5-2630 v4 2.2GHz, 25M Cache, 8.0 GT/s QPI, Turbo, HT, 10C/20T (85W) Max Mem 2133MHz
	* 128 GB Ram
	* 1 TB Disk
  
## My approach to achive 0.99046

FineTuning: For all of my final predictions I used finetuning with the same base model: 
Inception-v3 pre-trained on Imagenet. I tried other pre-trained models like VGG16, ResNet50, etc. 
but none performed nearly as well as Inception so I didn't use them.

Image Sizing: The images in the dataset were size 1154px x 866px. 
I ended up using a wide range of image sizes 400x400, 500x500, 600x600, 700x700, and 800x800. 
I think the combination of all of them all helped to achieve the final score.

Cross Validation: 5-fold cross-validation. 
So each fold I ran predictions on the test set and then took the average of 5 folds.

Batch Generator: Due to the larger image sizes and because I used 5-fold CV for training, 
there was no way to load all of the images into memory. 
The solution was that I created a batch generator for all my datasets including training, validation, and test set to load the images in batches.

Data Augmentation: Data augmentation is very important to avoid overfitting. 
I created some simple random augmentations like horizontal/vertical flips and rotations using numpy. 
I tried other augmentations like shift, zoom and shearing but all of these produced lower scores so I ended up not using them.

Semi-Supervised Learning: This was a very small training set (2295 images), 
so I felt that the model could improve a lot from semi-supervised learning so I used a technique called pseudo labeling.

Ensembling: It is very important to average predictions to help the models generalize 
and to avoid any bias because each trained model will potentially pick up on different information.
 My final submission included averages from 19 different predictions and most of those predictions were also 5-fold averages from CV.

# Code

Version 1: cnn_v1.py - Image size 128X128 - Naive Bagging (8 ANN) - Random rotation, shift and horizontal and vertical flip of images (LB ~0.95)

Version 2: cnn_v2.py - Increased the Image size to 190X256 (LB ~0.96)

Version 3: cnn_v3.py - Increased the Image size to 256X356 - Increased the momentum in SGD (LB ~0.97600)

Version 4: cnn_v4.py - New CNN structure. (LB ~0.97889)

Version 5: cnn_v5.py - Using pre-trained Inception-V3 CNN. Allows to get 0.99046 on LB. (LB ~0.99046)
