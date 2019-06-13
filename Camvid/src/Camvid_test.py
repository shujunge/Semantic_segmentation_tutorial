import os
import numpy as np
import keras.backend as K
import pandas as pd
from src.segnet import *
from keras.losses import *
from src.my_losses import *
from keras.models import *
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
plt.switch_backend('agg')


model_name="Camvid_segnet"
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

    test_data = []
    test_label = []
    src_images=[]
    src_labels=[]

    with open(path + 'test.txt') as f:
      test_txt = f.readlines()
      test_txt = np.array([line.split(' ') for line in test_txt])
    for i in tqdm(range(len(test_txt))):
      image = cv2.imread(path + test_txt[i][0][15:])
      label = cv2.imread( path + test_txt[i][1][15:][:-1])[:,:,0]
      src_images.append(cv2.resize(image,data_shape[::-1]))
      src_labels.append(cv2.resize(label,data_shape[::-1]))
      test_data.append(normalized(cv2.resize(image,data_shape[::-1])))
      test_label.append(binarylab(cv2.resize(label,data_shape[::-1])))

    return np.array(src_images),np.array(src_labels),np.array(test_data), np.array(test_label)


def IoU(y_true, y_pred, eps=1e-6):
  if K.max(y_true) == 0.0:
    return IoU(1 - y_true, 1 - y_pred)
  intersection = K.sum(y_true * y_pred, axis=[1, 2, 3])
  union = K.sum(y_true, axis=[1, 2, 3]) + K.sum(y_pred, axis=[1, 2, 3]) - intersection
  return K.mean((intersection + eps) / (union + eps), axis=0)




Sky = [128,128,128]
Building = [128,0,0]
Pole = [192,192,128]
Road_marking = [255,69,0]
Road = [128,64,128]
Pavement = [60,40,222]
Tree = [128,128,0]
SignSymbol = [192,128,128]
Fence = [64,64,128]
Car = [64,0,128]
Pedestrian = [64,64,0]
Bicyclist = [0,128,192]
Unlabelled = [0,0,0]

label_colours = np.array([Sky, Building, Pole, Road, Pavement,
                          Tree, SignSymbol, Fence, Car, Pedestrian, Bicyclist, Unlabelled])

def visualize(temp, plot=True):
    r = temp.copy()
    g = temp.copy()
    b = temp.copy()
    for l in range(0,11):
        r[temp==l]=label_colours[l,0]
        g[temp==l]=label_colours[l,1]
        b[temp==l]=label_colours[l,2]

    rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
    rgb[:,:,0] = (r/255.0)
    rgb[:,:,1] = (g/255.0)
    rgb[:,:,2] = (b/255.0)
    if plot:
        plt.imshow(rgb)
    else:
        return rgb



if __name__=="__main__":
  
  loss_names=["dice_coef_loss"]
  for loss_name in loss_names:
    name="segnet"+ "_" + loss_name
    autoencoder =load_model('./weights/%s_weights.h5'%(name),
                            custom_objects={'IoU':IoU,"dice_coef_loss":dice_coef_loss})
    autoencoder.compile(loss=eval(loss_name), optimizer='adadelta',metrics=[IoU])
  
    src_images,src_labels,test_images,test_labels = prep_data()
    print("loading done!")
    print(test_images.shape,test_labels.shape)
    print(autoencoder.evaluate(test_images,test_labels))
    y_= autoencoder.predict(test_images)
    print(K.eval(IoU(y_.astype(np.float16),test_labels.astype(np.float16))))
  
    output = autoencoder.predict_proba(test_images[0:1])
    print(output.shape)
    plt.figure(figsize=(8, 8))
    pred = visualize(np.argmax(output[0], axis=-1).reshape((360, 480)), False)
    plt.subplot(121)
    plt.imshow(pred * 255)
    plt.subplot(122)
    plt.imshow(src_labels[0])
    plt.savefig("results.png",dpi=560)
    exit()
    
    
    
    for num_index in range(8):
      plt.figure(figsize=(8, 8))
      plt.imshow(src_images[num_index].reshape(*data_shape, 3))
      plt.title("val_images")
      plt.savefig("results/%d_images.png" % num_index, dpi=560)
      for j in range(12):
        plt.figure(figsize=(8, 8))
        plt.subplot(121)
        plt.imshow(y_[num_index, :, :, j].reshape(*data_shape))
        plt.title("predicted")
        plt.subplot(122)
        plt.imshow(test_labels[num_index, :, :, j].reshape(*data_shape))
        plt.title("label")
        plt.savefig("results/%d_%d_results.png" % (num_index, j), dpi=560)
    

