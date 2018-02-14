# Kaggle-Invasive Species Monitoring
My solutions that uses Keras (81st/513 place LB 0.99046)

## Environment
	* 2 Intel Xeon E5-2630 v4 2.2GHz, 25M Cache, 8.0 GT/s QPI, Turbo, HT, 10C/20T (85W) Max Mem 2133MHz
	* 128 GB Ram
	* 1 TB Disk
  
## My approach to achieve 0.99046

**Image Sizing:** The images in the dataset were size 1154px x 866px. 
I used a small range of image sizes 128X128, 190X256 and 256X356. 
If I used a greater size I cloud have a better final score, but my environment doesn't have GPU.

**Cross Validation:** 8-fold cross-validation. 
So each fold I ran predictions on the test set and then took the average of 8 folds.

**Data Augmentation:** Data augmentation is very important to avoid overfitting. 
I used some random augmentations like horizontal/vertical flips and rotations and I tried other augmentations like shift and zoom.

**Ensembling:** It is very important to average predictions to help the models generalize 
and to avoid any bias because each trained model will potentially pick up on different information.

**FineTuning:** At the end I used finetuning with the same base model Inception-v3 pre-trained on Imagenet.

## Code

**Version 1:** cnn_v1.py - Image size 128X128 - Naive Bagging (8 ANN) - Random rotation, shift and horizontal and vertical flip of images (LB ~0.95)

**Version 2:** cnn_v2.py - Increased the Image size to 190X256 (LB ~0.96)

**Version 3:** cnn_v3.py - Increased the Image size to 256X356 - Increased the momentum in SGD (LB ~0.97600)

**Version 4:** cnn_v4.py - New CNN structure. (LB ~0.97889)

**Version 5:** cnn_v5.py - Using pre-trained Inception-V3 CNN. Allows to get 0.99046 on LB. (LB ~0.99046)

## References
[Kaggle Invasive Species Monitoring](https://www.kaggle.com/c/invasive-species-monitoring)
