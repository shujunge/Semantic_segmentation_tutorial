import keras.backend as K
import numpy as np


def IoU(y_true, y_pred, eps=1e-6):
  y =y_true[:,:,:,1:]
  y_ = y_pred[:,:,:,1:]
  intersection = K.sum(y * y_, axis=[1, 2])
  union = K.sum(y, axis=[1, 2]) + K.sum(y_, axis=[1, 2]) - intersection
  return K.mean((intersection + eps) / (union + eps),axis=0)


y=np.zeros((2,224,224,21))
y[0,30:40,50:80,0]=1
y[0,30:40,50:80,1]=1
y[0,10:40,60:80,4]=1
y[0,30:60,50:90,5]=1
y_ = y.copy()
y_[0,12:42,62:82,4]=1
y_[0,20:50,50:90,5]=1
print(K.eval(IoU(y,y_)))