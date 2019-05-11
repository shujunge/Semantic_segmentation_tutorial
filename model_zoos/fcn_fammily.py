from keras.models import *
from keras.layers import *


def FCN32v1(nClasses, input_shape=(224, 224, 1)):
  # assert input_height % 32 == 0
  # assert input_width % 32 == 0
  # https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_th_dim_ordering_th_kernels.h5
  img_input = Input(shape=input_shape)
  
  x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
  x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
  x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)
  f1 = x
  # Block 2
  x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
  x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
  x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)
  f2 = x
  
  # Block 3
  x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
  x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
  x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
  x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)
  f3 = x
  
  # Block 4
  x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
  x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
  x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
  x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)
  f4 = x
  
  # Block 5
  x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
  x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
  x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
  x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)
  f5 = x
  
  o = f5
  
  o = (Conv2D(1024, (7, 7), activation='relu', padding='same'))(o)
  o = Dropout(0.5)(o)
  o = (Conv2D(1024, (1, 1), activation='relu', padding='same'))(o)
  o = Dropout(0.5)(o)
  
  o = (Conv2D(512, (1, 1), kernel_initializer='he_normal'))(o)
  
  o = Conv2DTranspose(256, kernel_size=(32, 32), strides=(32, 32), use_bias=False)(o)
  o = BatchNormalization()(o)
  o = (Activation('relu'))(o)
  x = Conv2D(64, (3, 3), padding='same')(o)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  # last conv
  x = Conv2D(nClasses, (3, 3), activation='sigmoid', padding='same')(x)
  model = Model(img_input, x)
  
  return model


def FCN32(nClasses, input_shape=(224, 224, 1)):
  img_input = Input(shape=input_shape)
  
  x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
  x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
  x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)
  f1 = x
  # Block 2
  x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
  x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
  x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)
  f2 = x
  
  # Block 3
  x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
  x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
  x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
  x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)
  f3 = x
  
  # Block 4
  x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
  x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
  x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
  x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)
  f4 = x
  
  # Block 5
  x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
  x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
  x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
  x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)
  f5 = x
  
  o = Conv2D(
    filters=4096,
    kernel_size=(
      7,
      7),
    padding="same",
    activation="relu",
    name="fc6")(
    f5)
  o = Dropout(rate=0.5)(o)
  o = Conv2D(
    filters=4096,
    kernel_size=(
      1,
      1),
    padding="same",
    activation="relu",
    name="fc7")(o)
  o = Dropout(rate=0.5)(o)
  
  o = Conv2D(filters=nClasses, kernel_size=(1, 1), padding="same", activation="relu", kernel_initializer="he_normal",
             name="score_fr")(o)
  
  o = Conv2DTranspose(filters=nClasses, kernel_size=(32, 32), strides=(32, 32), padding="valid", activation=None,
                      name="score2")(o)
  
  o = Activation("sigmoid")(o)
  
  fcn8 = Model(inputs=img_input, outputs=o)
  return fcn8


def FCN8_helper(nClasses, input_shape=(224, 224, 1)):
  # assert input_height % 32 == 0
  # assert input_width % 32 == 0
  
  img_input = Input(shape=input_shape)
  
  x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
  x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
  x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)
  f1 = x
  # Block 2
  x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
  x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
  x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)
  f2 = x
  
  # Block 3
  x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
  x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
  x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
  x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)
  f3 = x
  
  # Block 4
  x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
  x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
  x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
  x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)
  f4 = x
  
  # Block 5
  x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
  x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
  x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
  x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)
  f5 = x
  
  o = Conv2D(
    filters=2048,
    kernel_size=(
      7,
      7),
    padding="same",
    activation="relu",
    name="fc6")(
    f5)
  o = Dropout(rate=0.5)(o)
  o = Conv2D(
    filters=2048,
    kernel_size=(
      1,
      1),
    padding="same",
    activation="relu",
    name="fc7")(o)
  o = Dropout(rate=0.5)(o)
  
  o = Conv2D(filters=nClasses, kernel_size=(1, 1), padding="same", activation="relu", kernel_initializer="he_normal",
             name="score_fr")(o)
  
  o = Conv2DTranspose(filters=nClasses, kernel_size=(2, 2), strides=(2, 2), padding="valid", activation=None,
                      name="score2")(o)
  
  fcn8 = Model(inputs=img_input, outputs=o)
  
  return fcn8


def FCN8(nClasses, input_shape=(224, 224, 1)):
  fcn8 = FCN8_helper(128, input_shape)
  # Conv to be applied on Pool4
  skip_con1 = Conv2D(128, kernel_size=(1, 1), padding="same", activation=None, kernel_initializer="he_normal",
                     name="score_pool4")(fcn8.get_layer("block4_pool").output)
  Summed = add(inputs=[skip_con1, fcn8.output])
  
  x = Conv2DTranspose(256, kernel_size=(2, 2), strides=(2, 2), padding="valid", activation=None,
                      name="score4")(Summed)
  
  ###
  skip_con2 = Conv2D(256, kernel_size=(1, 1), padding="same", activation=None, kernel_initializer="he_normal",
                     name="score_pool3")(fcn8.get_layer("block3_pool").output)
  Summed2 = add(inputs=[skip_con2, x])
  
  #####
  Up = Conv2DTranspose(64, kernel_size=(8, 8), strides=(8, 8),
                       padding="valid", activation=None, name="upsample")(Summed2)
  
  o = BatchNormalization()(Up)
  o = (Activation('relu'))(o)
  x = Conv2D(32, (3, 3), padding='same')(o)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  # last conv
  x = Conv2D(nClasses, (3, 3), activation='sigmoid', padding='same')(x)
  
  mymodel = Model(inputs=fcn8.input, outputs=x)
  
  return mymodel