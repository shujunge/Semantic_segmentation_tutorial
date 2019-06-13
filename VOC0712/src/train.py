import h5py
import numpy as np
from tqdm import tqdm

from keras.callbacks import *
from keras.optimizers import *
from unet import *


def load_h5(path):
  print('loading', path)
  file = h5py.File(name=path, mode='r')
  return file['images'][:], file['labels'][:]


def dice_coef_loss(y_true, y_pred):
  def dice_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(y_true * y_pred, axis=[1, 2, 3])
    union = K.sum(y_true, axis=[1, 2, 3]) + K.sum(y_pred, axis=[1, 2, 3])
    return K.mean((2. * intersection + smooth) / (union + smooth), axis=0)
  
  return 1 - dice_coef(y_true, y_pred, smooth=1)


def IoU(y_true, y_pred, eps=1e-6):
  if K.max(y_true) == 0.0:
    return IoU(1 - y_true, 1 - y_pred)  ## empty image; calc IoU of zeros
  intersection = K.sum(y_true * y_pred, axis=[1, 2, 3])
  union = K.sum(y_true, axis=[1, 2, 3]) + K.sum(y_pred, axis=[1, 2, 3]) - intersection
  return K.mean((intersection + eps) / (union + eps), axis=0)



train_images, train_labels = load_h5("../dataset/voc2012_train.h5")
print("traing shape:", train_images.shape, train_labels.shape)
val_images, val_labels = load_h5("../dataset/voc2012_val.h5")
print("valid shape:", val_images.shape, val_labels.shape)

from keras.models import *
model = unet(21, (224, 224, 3))
loss_names =["dice_coef_loss"]# ["binary_crossentropy" ,"dice_coef_loss"]
for loss_name in loss_names:
  model.compile(loss=eval(loss_name), optimizer=Adam(lr=0.03), metrics=[IoU])
  checkpoint = ModelCheckpoint('./weights/sigmoid_%s_weights.h5' % loss_name,  # model filename
                               monitor='val_IoU',  # quantity to monitor
                               verbose=1,  # verbosity - 0 or 1
                               save_best_only=True,  # The latest best model will not be overwritten
                               mode='max')  # The decision to overwrite model is m
  model= Sequential()
  history = model.fit(train_images,train_labels,batch_size=8,shuffle=True,validation_data=(val_images,val_labels)
                      ,epochs=100, verbose=1,callbacks=[checkpoint])
  
  import pandas as pd
  
  pf = pd.DataFrame(history.history)
  pf.to_csv("./training_history/sigmoid_%s_history.csv" % loss_name, index=False)

