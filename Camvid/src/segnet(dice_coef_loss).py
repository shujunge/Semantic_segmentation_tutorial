

import matplotlib.cm as cm
import itertools
import numpy as np
np.random.seed(1337) # for reproducibility
import os
from keras.models import *
from keras.layers import *
from keras.utils import *
from keras.utils import plot_model
import keras.backend as K
import cv2
from tqdm import tqdm
path = '../CamVid/'
data_shape = 360*480

def normalized(rgb):
    #return rgb/255.0
    norm=np.zeros((rgb.shape[0], rgb.shape[1], 3),np.float32)

    b=rgb[:,:,0]
    g=rgb[:,:,1]
    r=rgb[:,:,2]

    norm[:,:,0]=cv2.equalizeHist(b)
    norm[:,:,1]=cv2.equalizeHist(g)
    norm[:,:,2]=cv2.equalizeHist(r)

    return norm

def binarylab(labels):
    x = np.zeros([360,480,12])
    for i in range(360):
        for j in range(480):
            x[i,j,labels[i][j]]=1
    return x


import keras
class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs,  batch_size=32, dim=(32,32), n_channels=1,
                 n_classes=12, shuffle=False):
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.zeros((self.batch_size,*self.dim,self.n_classes))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
          X[i,:]=normalized(cv2.imread(path + ID[0][15:]))
          y[i,:] = binarylab(cv2.imread(path + ID[1][15:][:-1])[:,:,0])

#         y=y.reshape(self.batch_size,data_shape,self.n_classes)
        return X, y

params = {'dim': (360,480),
          'batch_size': 4,  
          'n_classes': 12,
          'n_channels': 3,
          'shuffle': True} 
with open(path+'train.txt') as f:
    train_txt = f.readlines()
    train_txt = np.array([line.split(' ') for line in train_txt])
with open(path+'test.txt') as f:
  test_txt = f.readlines()
  test_txt = np.array([line.split(' ') for line in test_txt])
import random
index = [i for i in range(len(train_txt))]  
random.shuffle(index) 
train_txt =train_txt[index]
print(train_txt.shape)
print(test_txt.shape)

training_generator = DataGenerator(train_txt,  **params)
validation_generator = DataGenerator(test_txt,  **params)

class_weighting= [0.2595, 0.1826, 4.5640, 0.1417, 0.5051, 0.3826, 9.6446, 1.8418, 6.6823, 6.2478, 3.0, 7.3614]



def dice_coef_loss(y_true, y_pred):
  def dice_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(y_true * y_pred, axis=[1, 2, 3])
    union = K.sum(y_true, axis=[1, 2, 3]) + K.sum(y_pred, axis=[1, 2, 3])
    return K.mean((2. * intersection + smooth) / (union + smooth), axis=0)
  return 1 - dice_coef(y_true, y_pred, smooth=1)

def IoU(y_true, y_pred, eps=1e-6):
    if K.max(y_true) == 0.0:
        return IoU(1-y_true, 1-y_pred) ## empty image; calc IoU of zeros
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3]) - intersection
    return K.mean( (intersection + eps) / (union + eps), axis=0)

from segnet import *
autoencoder = segnet(nclasses=12, input_shape=(360,480, 3))

autoencoder.compile(loss=dice_coef_loss, optimizer='adadelta',metrics=[IoU])


nb_epoch = 20
from keras.callbacks import *
checkpoint = ModelCheckpoint('weights.h5',  # model filename
                             monitor='val_IoU', # quantity to monitor
                             verbose=1, # verbosity - 0 or 1
                             save_best_only= True, # The latest best model will not be overwritten
                             mode='max') # The decision to overwrite model is m

history = autoencoder.fit_generator(generator=training_generator,validation_data=validation_generator,
                                    epochs=nb_epoch,verbose=1,callbacks=[checkpoint], class_weight=class_weighting)
##without  mIOU:0.57700
