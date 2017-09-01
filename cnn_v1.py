import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import re

import math as mt

from sklearn import model_selection
from sklearn import metrics

import keras
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense , Dropout , Flatten
from keras.layers import MaxPooling2D
from keras.layers.convolutional import Conv2D
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
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
for img_path in sorted(os.listdir(train_data_dir),key=lambda x: (int(re.sub('\D','',x)),x)): # sort by key
    img = image.load_img(train_data_dir+'/'+img_path,target_size=(128, 128))
    train_img.append(image.img_to_array(img))
# check shape
print('Train len: {}'.format(len(train_img)))

# I have to open the img in ordered way
for img_path in sorted(os.listdir(test_data_dir),key=lambda x: (int(re.sub('\D','',x)),x)): # sort by key
    img = image.load_img(test_data_dir+'/'+img_path,target_size=(128, 128))
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
# Convolutional Neural Network 
# Accuracy: 97% - LB: 95%
#########################################################

def model_nn():
    # Create the model
    model = Sequential()
    model.add(Conv2D(16, (3, 3), activation='relu', input_shape=(3,128, 128)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(2048, activation='relu'))
    model.add(Dropout(0.65))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.55))
    model.add(Dense(1, activation='sigmoid'))
    sgd = optimizers.SGD(lr = 0.001, decay = 1e-6, momentum = 0.8, nesterov = True)
    model.compile(loss = 'binary_crossentropy', optimizer = sgd, metrics=['accuracy'])
    print(model.summary())
    return model

# build the model
model = model_nn()

# KFold
n_fold = 8
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
            rotation_range = 30,#Degree range for random rotations
            width_shift_range = 0.2,#Range for random horizontal shifts
            height_shift_range = 0.2,#Range for random vertical shifts
            # zca_whitening = True,
            shear_range = 0.2,
            zoom_range = 0.2,#Range for random zoom
            horizontal_flip = True,
            vertical_flip = True,
            fill_mode = 'nearest')
        datagen.fit(x_tr)

        model = model_nn()
        earlystop = keras.callbacks.EarlyStopping(monitor='val_loss', patience = 15, verbose=0, mode='auto')
        model.fit_generator(datagen.flow(x_tr, y_tr, batch_size = 64),
            validation_data = (x_te, y_te), callbacks = [earlystop],
            steps_per_epoch = len(train_img) / 64, epochs = 1000, verbose = 2)

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
