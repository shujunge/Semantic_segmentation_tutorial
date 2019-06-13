import h5py
import numpy as np
from tqdm import tqdm

import keras
from keras.callbacks import *
from keras.optimizers import *
from keras.models import *
from unet import *


def load_h5(path):
  print('loading', path)
  file = h5py.File(name=path, mode='r')
  return file['images'], file['labels']



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


loss_name = "binary_crossentropy"

model=unet(21,(224,224,3))
model.load_weights('./weights/sigmoid_%s_weights.h5' % loss_name)
model.compile(loss=loss_name, optimizer="adadelta", metrics=[IoU])

y_pre = model.predict(val_images[:20])
y_pre =y_pre>0.5
y_pre.astype(np.int8)
y_pre = np.sum(y_pre, axis=-1)
y_pre/=y_pre.max()
y_pre*=255

val_labels = val_labels[:-1, :, :]
print("y_pre:", y_pre.shape)
print(val_labels.shape)

import matplotlib.pyplot as plt
plt.switch_backend('agg')
# for num_index in tqdm(range(20)):
#   plt.figure(figsize=(8, 8))
#   plt.subplot(131)
#   plt.imshow(val_images[num_index].reshape(224, 224, 3))
#   plt.title("images")
#   plt.subplot(132)
#   plt.imshow(y_pre[num_index].reshape(224, 224))
#   plt.title("predicted")
#   plt.subplot(133)
#   plt.imshow(val_labels[num_index].reshape(224, 224))
#   plt.title("ground_truth")
#   plt.savefig("results/%d_results.png" % num_index, dpi=560)


# for num_index in tqdm(range(20)):
#   plt.figure(figsize=(8, 8))
#   plt.subplot(121)
#   plt.imshow(val_images[num_index].reshape(224, 224, 3))
#   plt.title("images")
#   plt.subplot(122)
#   plt.imshow(val_labels[num_index].reshape(224, 224))
#   plt.title("ground_truth")
#   plt.savefig("results/%d_results.png" % num_index, dpi=560)
#   for j in range(20):
#     plt.figure(figsize=(8, 8))
#     plt.imshow(y_pre[num_index,:,:,j].reshape(224, 224))
#     plt.title("predicted")
#     plt.savefig("results/%d_%d_results.png" % (num_index,j), dpi=560)

# for images,label in val_generator:
#   for num_index in tqdm(range(8)):
#     plt.figure(figsize=(8, 8))
#     plt.subplot(121)
#     plt.imshow(images[num_index].reshape(224, 224, 3))
#     plt.title("images")
#     plt.savefig("results/%d_images.png" % (num_index), dpi=560)
#     for j in range(21):
#       plt.figure(figsize=(8, 8))
#       plt.imshow(label[num_index,:,:,j].reshape(224, 224))
#       plt.title("predicted")
#       plt.savefig("results/%d_%d_results.png" % (num_index,j), dpi=560)




import h5py
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import keras.backend as K
from keras.callbacks import *
from keras.optimizers import *
from keras.models import *
from keras.losses import *
from unet import *


def load_h5(path):
  print('loading', path)
  file = h5py.File(name=path, mode='r')
  return file['images'][:], file['labels'][:]

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

loss_name = "binary_crossentropy"
model = unet(21, (224, 224, 3))
model.compile(loss=eval(loss_name), optimizer=Adam(lr=0.03), metrics=[IoU])
model.load_weights('./weights/sigmoid_%s_weights.h5' % loss_name)

y_pre = model.predict(val_images)
print("y_pre type:",y_pre.dtype)
print(K.eval(IoU(val_labels.astype(np.float16),y_pre.astype(np.float16))))
for num_index in range(8):
  plt.figure(figsize=(8, 8))
  plt.imshow(val_images[num_index].reshape(224, 224, 3))
  plt.title("val_images")  
  plt.savefig("results/%d_images.png" % num_index, dpi=560)
  for j in range(21):
    plt.figure(figsize=(8, 8))
    plt.subplot(121)
    plt.imshow(y_pre[num_index,:,:,j].reshape(224, 224))
    plt.title("predicted")
    plt.subplot(122)
    plt.imshow(val_labels[num_index,:,:,j].reshape(224, 224))
    plt.title("label")
    plt.savefig("results/%d_%d_results.png" % (num_index,j), dpi=560)

