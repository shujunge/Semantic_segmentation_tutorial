from keras.layers import *
from keras.models import Sequential

import warnings

warnings.filterwarnings('ignore')


def get_frontend(input_shape) -> Sequential:
  model = Sequential()
  # model.add(ZeroPadding2D((1, 1), input_shape=(input_width, input_height, 3)))
  model.add(Conv2D(64, (3, 3), activation='relu', padding='same', name='conv1_1', input_shape=input_shape))
  model.add(Conv2D(64, (3, 3), activation='relu', padding='same', name='conv1_2'))
  model.add(MaxPooling2D((2, 2), strides=(2, 2)))
  
  model.add(Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2_1'))
  model.add(Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2_2'))
  model.add(MaxPooling2D((2, 2), strides=(2, 2)))
  
  model.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_1'))
  model.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_2'))
  model.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_3'))
  model.add(MaxPooling2D((2, 2), strides=(2, 2)))
  
  model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_1'))
  model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_2'))
  model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_3'))
  
  # Compared to the original VGG16, we skip the next 2 MaxPool layers,
  # and go ahead with dilated convolutional layers instead
  
  model.add(AtrousConvolution2D(512, 3, 3, atrous_rate=(2, 2), activation='relu', name='conv5_1'))
  model.add(AtrousConvolution2D(512, 3, 3, atrous_rate=(2, 2), activation='relu', name='conv5_2'))
  model.add(AtrousConvolution2D(512, 3, 3, atrous_rate=(2, 2), activation='relu', name='conv5_3'))
  
  # # Compared to the VGG16, we replace the FC layer with a convolution
  #
  # model.add(AtrousConvolution2D(4096,7,7, atrous_rate=(4, 4), activation='relu', name='fc6'))
  model.add(Dropout(0.5))
  model.add(Conv2D(256, (1, 1), activation='relu', name='fc7'))
  model.add(Dropout(0.5))
  # Note: this layer has linear activations, not ReLU
  model.add(Conv2D(256, (1, 1), activation='linear', name='fc-final'))
  
  # model.layers[-1].output_shape == (None, 16, 16, 21)
  return model


def add_context(model: Sequential) -> Sequential:
  """ Append the context layers to the frontend. """
  # model.add(ZeroPadding2D(padding=(33, 33)))
  # model.add(Conv2D(42, (3, 3), activation='relu', name='ct_conv1_1'))
  # model.add(Conv2D(42, (3, 3), activation='relu', name='ct_conv1_2'))
  # model.add(AtrousConvolution2D(84, 3, 3, atrous_rate=(2, 2), activation='relu', name='ct_conv2_1'))
  # model.add(AtrousConvolution2D(168, 3, 3, atrous_rate=(4, 4), activation='relu', name='ct_conv3_1'))
  # model.add(AtrousConvolution2D(336, 3, 3, atrous_rate=(8, 8), activation='relu', name='ct_conv4_1'))
  # model.add(AtrousConvolution2D(672, 3, 3, atrous_rate=(16, 16), activation='relu', name='ct_conv5_1'))
  # model.add(Conv2D(672, (3, 3), activation='relu', name='ct_fc1'))
  model.add(Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same'))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  
  model.add(Conv2D(256, (3, 3), padding='same'))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  
  model.add(Conv2DTranspose(128, (7, 7), strides=(7, 7), padding='same'))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  
  model.add(Conv2D(64, (3, 3), padding='same'))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  
  # last conv
  model.add(Conv2D(1, (3, 3), activation='sigmoid', padding='same'))
  
  return model


def context(model: Sequential) -> Sequential:
  """ Append the context layers to the frontend. """
  model.add(ZeroPadding2D(padding=(33, 33)))
  model.add(Conv2D(42, (3, 3), activation='relu', name='ct_conv1_1'))
  model.add(Conv2D(42, (3, 3), activation='relu', name='ct_conv1_2'))
  model.add(AtrousConvolution2D(84, 3, 3, atrous_rate=(2, 2), activation='relu', name='ct_conv2_1'))
  model.add(AtrousConvolution2D(168, 3, 3, atrous_rate=(4, 4), activation='relu', name='ct_conv3_1'))
  model.add(AtrousConvolution2D(336, 3, 3, atrous_rate=(8, 8), activation='relu', name='ct_conv4_1'))
  model.add(AtrousConvolution2D(672, 3, 3, atrous_rate=(16, 16), activation='relu', name='ct_conv5_1'))
  model.add(Conv2D(672, (3, 3), activation='relu', name='ct_fc1'))
  model.add(Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same'))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  
  model.add(Conv2D(256, (3, 3), padding='same'))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  
  model.add(Conv2DTranspose(128, (7, 7), strides=(7, 7), padding='same'))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  
  model.add(Conv2D(64, (3, 3), padding='same'))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  
  # last conv
  model.add(Conv2D(1, (3, 3), activation='sigmoid', padding='same'))
  
  return model



