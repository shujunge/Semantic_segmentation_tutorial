from keras.applications import imagenet_utils


def preprocess_input(x):
  return imagenet_utils.preprocess_input(x, mode='tf')


def iou_score(gt, pr, class_weights=1., smooth=1.0, per_image=True, threshold=None):
  if per_image:
    axes = [1, 2]
  else:
    axes = [0, 1, 2]
  
  if threshold is not None:
    pr = K.greater(pr, threshold)
    pr = K.cast(pr, K.floatx())
  
  intersection = K.sum(gt * pr, axis=axes)
  union = K.sum(gt + pr, axis=axes) - intersection
  iou = (intersection + smooth) / (union + smooth)
  # mean per image
  if per_image:
    iou = K.mean(iou, axis=0)
  
  # weighted mean per class
  iou = K.mean(iou * class_weights)
  
  return iou


from keras.models import *
from keras.layers import *


def get_segnet_vgg16(inputs, n_classes):
  x = BatchNormalization()(inputs)
  
  # Block 1
  x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
  x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
  x = MaxPooling2D((2, 2), strides=(2, 2))(x)
  
  # Block 2
  x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
  x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
  x = MaxPooling2D((2, 2), strides=(2, 2))(x)
  
  # Block 3
  x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
  x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
  x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
  x = MaxPooling2D((2, 2), strides=(2, 2))(x)
  
  # Block 4
  x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
  x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
  x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
  x = MaxPooling2D((2, 2), strides=(2, 2))(x)
  
  # Block 5
  x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
  x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
  x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
  x = MaxPooling2D((2, 2), strides=(2, 2))(x)
  
  # Up Block 1
  x = UpSampling2D(size=(2, 2))(x)
  x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
  x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
  x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
  
  # Up Block 2
  x = UpSampling2D(size=(2, 2))(x)
  x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
  x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
  x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
  
  # Up Block 3
  x = UpSampling2D(size=(2, 2))(x)
  x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
  x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
  x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
  
  # Up Block 4
  x = UpSampling2D(size=(2, 2))(x)
  x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
  x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
  
  # Up Block 5
  x = UpSampling2D(size=(2, 2))(x)
  x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
  x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
  
  x = Conv2D(n_classes, (1, 1), activation='linear', padding='same')(x)
  
  return x

inputs =Input((224, 224, 3))
x = get_segnet_vgg16(inputs, 21)
x = Activation("softmax")(x)
model = Model(inputs,x)
# model.summary()