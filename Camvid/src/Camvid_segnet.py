import os
import numpy as np
import keras.backend as K
import pandas as pd
from src.segnet import *
from keras.losses import *
from src.my_losses import *
from keras.callbacks import *
import cv2
import random
from tqdm import tqdm

path = '../CamVid/'
data_shape = (360,480)

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
    x = np.zeros([*data_shape,12])
    for i in range(data_shape[0]):
        for j in range(data_shape[1]):
            x[i,j,labels[i][j]]=1
    return x

def prep_data():
    trainval_data = []
    trainval_label = []
    test_data = []
    test_label = []
    with open(path+'train.txt') as f:
      train_txt = f.readlines()
      train_txt = [line.split(' ') for line in train_txt]
    with open(path+'val.txt') as f:
        txt1 = f.readlines()
        train_txt += [line.split(' ') for line in txt1]
    train_txt = np.array(train_txt)
    index = [i for i in range(len(train_txt))]
    random.shuffle(index)
    train_txt = train_txt[index]
    for i in tqdm(range(len(train_txt))):
      image = cv2.imread(path + train_txt[i][0][15:])
      trainval_data.append( normalized( cv2.resize(image, data_shape[::-1]) ) )
      label =cv2.imread( path + train_txt[i][1][15:][:-1])[:,:,0]
      trainval_label.append( binarylab( cv2.resize(label, data_shape[::-1]) ) )

    
    with open(path + 'test.txt') as f:
      test_txt = f.readlines()
      test_txt = np.array([line.split(' ') for line in test_txt])
    for i in tqdm(range(len(test_txt))):
      image = cv2.imread(path + test_txt[i][0][15:])
      test_data.append(normalized(cv2.resize(image, data_shape[::-1])))
      label = cv2.imread(path + test_txt[i][1][15:][:-1])[:, :, 0]
      test_label.append(binarylab(cv2.resize(label, data_shape[::-1])))

    return np.array(trainval_data), np.array(trainval_label),np.array(test_data), np.array(test_label)


def IoU(y_true, y_pred, eps=1e-6):
  if K.max(y_true) == 0.0:
    return IoU(1 - y_true, 1 - y_pred)
  intersection = K.sum(y_true * y_pred, axis=[1, 2, 3])
  union = K.sum(y_true, axis=[1, 2, 3]) + K.sum(y_pred, axis=[1, 2, 3]) - intersection
  return K.mean((intersection + eps) / (union + eps), axis=0)


if __name__=="__main__":
  
  train_data, train_label, test_images, test_labels = prep_data()
  train_label = np.reshape(train_label, (train_data.shape[0], *data_shape, 12))
  print("loading done!")
  print(train_data.shape, train_label.shape)
  class_weighting = [0.2595, 0.1826, 4.5640, 0.1417, 0.5051, 0.3826, 9.6446, 1.8418, 6.6823, 6.2478, 3.0, 7.3614]
  loss_names=["dice_coef_loss","categorical_crossentropy"]
  model_names=["segnet"]
  for model_name in model_names:
    for loss_name in loss_names:
      name = model_name + "_" + loss_name
      autoencoder =eval(model_name)(nclasses=12,input_shape=(*data_shape,3))
      autoencoder.compile(loss=eval(loss_name), optimizer='adadelta',metrics=[IoU])
      checkpoint = ModelCheckpoint('./weights/%s_weights.h5'%(name),
                                   monitor='val_IoU',
                                   verbose=1,
                                   save_best_only=True,
                                   mode='max')
      history = autoencoder.fit(train_data, train_label, batch_size=4, epochs=30,
                                validation_data=(test_images,test_labels),verbose=1,callbacks=[checkpoint] )
      pf = pd.DataFrame(history.history)
      pf.to_csv("./training_history/%s_history.csv"%(name), index=False)

      name = model_name+"_"+loss_name+"_"+"class_weighting"
      autoencoder = eval(model_name)(nclasses=12, input_shape=(*data_shape, 3))
      autoencoder.compile(loss=eval(loss_name), optimizer='adadelta', metrics=[IoU])
      checkpoint = ModelCheckpoint('./weights/%s_weights.h5' % (name),
                                   monitor='val_IoU',
                                   verbose=1,
                                   save_best_only=True,
                                   mode='max')
      history = autoencoder.fit(train_data, train_label, batch_size=4, epochs=30,
                                validation_data=(test_images, test_labels), verbose=1,
                                class_weight=class_weighting, callbacks=[checkpoint])
      pf = pd.DataFrame(history.history)
      pf.to_csv("./training_history/%s_history.csv" % (name), index=False)


