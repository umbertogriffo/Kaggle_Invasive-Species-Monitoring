import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import re

import math as mt

from sklearn import model_selection
from sklearn import metrics

from scipy.misc import toimage
import matplotlib.pyplot as plt

import keras
from keras import optimizers
from keras.models import Model
from keras.layers import Dense , GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.applications import inception_v3
from keras import backend as K
K.set_image_dim_ordering('th')

output_path = '../Output/'
train_data_dir = '/home/mantica/Scrivania/InvasiveSpecies/train'
test_data_dir = '/home/mantica/Scrivania/InvasiveSpecies/test'
label_data_dir = '/home/mantica/Scrivania/InvasiveSpecies/train_labels.csv'

############################################
# load training and test data from directory
############################################

train_set = pd.read_csv(label_data_dir)

train_img, test_img = [], []
# I have to open the img in ordered way
print(len(os.listdir(train_data_dir)))
i = 0
for img_path in sorted(os.listdir(train_data_dir),key=lambda x: (int(re.sub('\D','',x)),x)): # sort by key
    img = image.load_img(train_data_dir+'/'+img_path,target_size=(256, 356))
    # img = image.load_img(train_data_dir+'/'+img_path)
    # Plot first 3 image
    if i in range(0, 3):
        imgOriginal = image.load_img(train_data_dir+'/'+img_path)
        
        plt.imshow(toimage(imgOriginal))
        plt.show()
        
        plt.imshow(toimage(img))
        plt.show()
        i = i + 1

    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.misc.imresize.html
    # https://graphicdesign.stackexchange.com/questions/26385/difference-between-none-linear-cubic-and-sinclanczos3-interpolation-in-image
    #img = imresize(img, (256,190), mode=None)#interp='cubic'
    train_img.append(image.img_to_array(img))
# check shape
print('Train len: {}'.format(len(train_img)))

# I have to open the img in ordered way
for img_path in sorted(os.listdir(test_data_dir),key=lambda x: (int(re.sub('\D','',x)),x)): # sort by key
    img = image.load_img(test_data_dir+'/'+img_path,target_size=(256, 356))
    #img = image.load_img(test_data_dir+'/'+img_path)
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.misc.imresize.html
    # https://graphicdesign.stackexchange.com/questions/26385/difference-between-none-linear-cubic-and-sinclanczos3-interpolation-in-image
    #img= imresize(img, (256,190), mode=None)#interp='cubic'
    test_img.append(image.img_to_array(img))
# check shape
print('Test len: {}'.format(len(test_img)))

# normalize inputs from 0-255 to 0.0-1.0
train_img = np.array(train_img, np.float32) / 255
train_label = np.array(train_set['invasive'].iloc[: ])
test_img = np.array(test_img, np.float32) / 255

"""Return the sample arithmetic mean of data."""
def mean(numbers):
    return float(sum(numbers)) / max(len(numbers), 1)
"""Return sum of square deviations of sequence data."""    
def sum_of_square_deviation(numbers,mean):
    return float(1/len(numbers) * sum((x - mean)** 2 for x in numbers)) 

#########################################################
# Inception V3 model, with weights pre-trained on ImageNet. 
#########################################################
epochs=1000
def model_nn():
    # Loading pre-trained model and adding custom layers
    base_model = inception_v3.InceptionV3(include_top=False, weights='imagenet', input_shape=(3,256, 356))
    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    x = Dense(1024, activation='relu')(x)
    p = Dense(1, activation='sigmoid')(x)
    
    model=Model(inputs=base_model.input,output=p)
    
    lrate = 0.0001
    decay = 1e-5
    sgd = optimizers.SGD(lr = lrate, decay = decay, momentum = 0.9, nesterov = True)
    model.compile(loss = 'binary_crossentropy', optimizer = sgd, metrics=['accuracy'])
    print(model.summary())
    return model

# Bagging using KFold with k = 5
n_fold = 5
kf = model_selection.KFold(n_splits = n_fold, shuffle = True)
eval_fun = metrics.roc_auc_score

def run_oof(tr_x, tr_y, te_x, kf):
    preds_train = np.zeros(len(tr_x), dtype = np.float)
    preds_test = np.zeros(len(te_x), dtype = np.float)
    train_loss = []; test_loss = []

    i = 1
    for train_index, test_index in kf.split(tr_x):
        x_tr = tr_x[train_index]; x_te = tr_x[test_index]
        y_tr = tr_y[train_index]; y_te = tr_y[test_index]

        datagen = ImageDataGenerator(
            # featurewise_center = True,
            rotation_range = 20,#Degree range for random rotations
            width_shift_range = 0.2,#Range for random horizontal shifts
            height_shift_range = 0.2,#Range for random vertical shifts
            # zca_whitening = True,
            shear_range = 0.2,
            zoom_range = 0.2,#Range for random zoom
            horizontal_flip = True,
            vertical_flip = True,
            fill_mode = 'nearest')
        datagen.fit(x_tr)
        
        # Build the model
        model = model_nn()
        earlystop = keras.callbacks.EarlyStopping(monitor='val_loss', patience = 15, verbose=0, mode='auto')
        
        model.fit_generator(datagen.flow(x_tr, y_tr, batch_size = 16),
            validation_data = (x_te, y_te), 
            callbacks = [earlystop],
            steps_per_epoch = len(train_img) / 16,
            epochs = epochs, verbose = 2)

        train_loss.append(eval_fun(y_tr, model.predict(x_tr)[:, 0]))
        test_loss.append(eval_fun(y_te, model.predict(x_te)[:, 0]))

        preds_train[test_index] = model.predict(x_te)[:, 0]
        # sum the predicted values
        preds_test += model.predict(te_x)[:, 0]
        
        #save results
        output = pd.read_csv('/home/mantica/Scrivania/InvasiveSpecies/sample_submission.csv')
        output['invasive'] = preds_test
        output.to_csv(output_path+str(test_loss[-1])+'_submission.csv', index=False)

        print('{0}: Train {1:0.5f} Val {2:0.5f}'.format(i, train_loss[-1], test_loss[-1]))
        i += 1

    print('Train: ', train_loss)
    print('Val: ', test_loss)

    mean_train_loss = mean(train_loss)
    mean_test_loss = mean(test_loss)
    standard_deviation_train_loss = mt.sqrt(sum_of_square_deviation(train_loss,mean_train_loss))
    standard_deviation_test_loss = mt.sqrt(sum_of_square_deviation(test_loss,mean_test_loss))
    print('Mean Train {0:0.5f}_Test {1:0.5f}\n\n'.format(mean_train_loss, mean_test_loss))
    print('Stdev Train {0:0.5f}_Test {1:0.5f}\n\n'.format(standard_deviation_train_loss, standard_deviation_test_loss))
    
    # average the predicted values
    preds_test /= n_fold
    return preds_train, preds_test

train_pred, test_pred = run_oof(train_img, train_label, test_img, kf)

test_set = pd.read_csv('/home/mantica/Scrivania/InvasiveSpecies/sample_submission.csv')
test_set['invasive'] = test_pred
test_set.to_csv(output_path+'submit_mean.csv', index = None)
