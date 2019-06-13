from keras.models import Model
from keras.layers import *
from keras import backend as K
from keras.applications.resnet50 import ResNet50
import warnings

warnings.filterwarnings('ignore')


def att_modle(g, x, F_int):
  g1 = Conv2D(F_int, 1, 1)(g)
  g1 = BatchNormalization()(g1)
  x1 = Conv2D(F_int, 1, 1)(x)
  x1 = BatchNormalization()(x1)
  psi = Add()([g1, x1])
  psi = Activation('relu')(psi)
  psi = Conv2D(F_int, 1, 1)(psi)
  psi = BatchNormalization()(psi)
  psi = Activation("sigmoid")(psi)
  out = Multiply()([x, psi])
  return out


def RCN(x, ch_out, t=2):
  for i in range(t):
    if i == 0:
      x1 = Conv2D(ch_out, (3, 3), padding='same')(x)
      x1 = BatchNormalization()(x1)
      x1 = Activation("relu")(x1)
    x1 = Add()([x, x1])
    x1 = Conv2D(ch_out, 3, padding='same')(x1)
    x1 = BatchNormalization()(x1)
    x1 = Activation("relu")(x1)
  
  return x1


def RCNN_block(x, ch_out, t=2):
  x = Conv2D(ch_out, 3, padding='same')(x)
  x1 = RCN(x, ch_out, t)
  x1 = RCN(x1, ch_out, t)
  x1 = Add()([x, x1])
  
  return x1


def unet(num_classes=1, input_shape=(224, 224, 1), vgg_weight_path=None):
  img_input = Input(input_shape)
  
  # Block 1
  x = Conv2D(64, (3, 3), padding='same', name='block1_conv1')(img_input)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  
  x = Conv2D(64, (3, 3), padding='same', name='block1_conv2')(x)
  x = BatchNormalization()(x)
  block_1_out = Activation('relu')(x)
  
  x = MaxPooling2D()(block_1_out)
  
  # Block 2
  x = Conv2D(128, (3, 3), padding='same', name='block2_conv1')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  
  x = Conv2D(128, (3, 3), padding='same', name='block2_conv2')(x)
  x = BatchNormalization()(x)
  block_2_out = Activation('relu')(x)
  
  x = MaxPooling2D()(block_2_out)
  
  # Block 3
  x = Conv2D(256, (3, 3), padding='same', name='block3_conv1')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  
  x = Conv2D(256, (3, 3), padding='same', name='block3_conv2')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  
  x = Conv2D(256, (3, 3), padding='same', name='block3_conv3')(x)
  x = BatchNormalization()(x)
  block_3_out = Activation('relu')(x)
  
  x = MaxPooling2D()(block_3_out)
  
  # Block 4
  x = Conv2D(512, (3, 3), padding='same', name='block4_conv1')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  
  x = Conv2D(512, (3, 3), padding='same', name='block4_conv2')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  
  x = Conv2D(512, (3, 3), padding='same', name='block4_conv3')(x)
  x = BatchNormalization()(x)
  block_4_out = Activation('relu')(x)
  
  x = MaxPooling2D()(block_4_out)
  
  # Block 5
  x = Conv2D(512, (3, 3), padding='same', name='block5_conv1')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  
  x = Conv2D(512, (3, 3), padding='same', name='block5_conv2')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  
  x = Conv2D(512, (3, 3), padding='same', name='block5_conv3')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  
  for_pretrained_weight = MaxPooling2D()(x)
  
  # Load pretrained weights.
  if vgg_weight_path is not None:
    vgg16 = Model(img_input, for_pretrained_weight)
    vgg16.load_weights(vgg_weight_path, by_name=True)
  
  # UP 1
  x = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  
  x = concatenate([x, block_4_out])
  x = Conv2D(512, (3, 3), padding='same')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  
  x = Conv2D(512, (3, 3), padding='same')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  
  # UP 2
  x = Conv2DTranspose(256, (3, 3), strides=(2, 2), padding='valid')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  
  x = concatenate([x, block_3_out])
  x = Conv2D(256, (3, 3), padding='same')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  
  x = Conv2D(256, (3, 3), padding='same')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  
  # UP 3
  x = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  
  x = concatenate([x, block_2_out])
  x = Conv2D(128, (3, 3), padding='same')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  
  x = Conv2D(128, (3, 3), padding='same')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  
  # UP 4
  x = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='valid')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  
  x = concatenate([x, block_1_out])
  x = Conv2D(64, (3, 3), padding='same')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  
  x = Conv2D(64, (3, 3), padding='same')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  
  # last conv
  x = Conv2D(num_classes, (3, 3), activation='sigmoid', padding='same')(x)
  
  model = Model(img_input, x)
  
  return model


def unet_small(num_classes=1, input_shape=(224, 224, 1), vgg_weight_path=None):
  img_input = Input(input_shape)
  
  # Block 1
  x = Conv2D(32, (3, 3), padding='same', name='block1_conv1')(img_input)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  
  x = Conv2D(32, (3, 3), padding='same', name='block1_conv2')(x)
  x = BatchNormalization()(x)
  block_1_out = Activation('relu')(x)
  
  x = MaxPooling2D()(block_1_out)
  
  # Block 2
  x = Conv2D(64, (3, 3), padding='same', name='block2_conv1')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  
  x = Conv2D(64, (3, 3), padding='same', name='block2_conv2')(x)
  x = BatchNormalization()(x)
  block_2_out = Activation('relu')(x)
  
  x = MaxPooling2D()(block_2_out)
  
  # Block 3
  x = Conv2D(128, (3, 3), padding='same', name='block3_conv1')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  
  x = Conv2D(128, (3, 3), padding='same', name='block3_conv2')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  
  x = Conv2D(128, (3, 3), padding='same', name='block3_conv3')(x)
  x = BatchNormalization()(x)
  block_3_out = Activation('relu')(x)
  
  x = MaxPooling2D()(block_3_out)
  
  # Block 4
  x = Conv2D(256, (3, 3), padding='same', name='block4_conv1')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  
  x = Conv2D(256, (3, 3), padding='same', name='block4_conv2')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  
  x = Conv2D(256, (3, 3), padding='same', name='block4_conv3')(x)
  x = BatchNormalization()(x)
  block_4_out = Activation('relu')(x)
  
  x = MaxPooling2D()(block_4_out)
  
  # Block 5
  x = Conv2D(256, (3, 3), padding='same', name='block5_conv1')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  
  x = Conv2D(256, (3, 3), padding='same', name='block5_conv2')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  
  x = Conv2D(256, (3, 3), padding='same', name='block5_conv3')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  
  for_pretrained_weight = MaxPooling2D()(x)
  
  # Load pretrained weights.
  if vgg_weight_path is not None:
    vgg16 = Model(img_input, for_pretrained_weight)
    vgg16.load_weights(vgg_weight_path, by_name=True)
  
  # UP 1
  x = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  
  x = concatenate([x, block_4_out])
  x = Conv2D(256, (3, 3), padding='same')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  
  x = Conv2D(256, (3, 3), padding='same')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  
  # UP 2
  x = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  
  x = concatenate([x, block_3_out])
  x = Conv2D(128, (3, 3), padding='same')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  
  x = Conv2D(128, (3, 3), padding='same')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  
  # UP 3
  x = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  
  x = concatenate([x, block_2_out])
  x = Conv2D(64, (3, 3), padding='same')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  
  x = Conv2D(64, (3, 3), padding='same')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  
  # UP 4
  x = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  
  x = concatenate([x, block_1_out])
  x = Conv2D(32, (3, 3), padding='same')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  
  x = Conv2D(16, (3, 3), padding='same')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  
  # last conv
  x = Conv2D(num_classes, (3, 3), activation='sigmoid', padding='same')(x)
  
  model = Model(img_input, x)
  
  return model


def Att_unet(num_classes=1, input_shape=(224, 224, 1), vgg_weight_path=None):
  img_input = Input(input_shape)
  
  # Block 1
  x = Conv2D(64, (3, 3), padding='same', name='block1_conv1')(img_input)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  
  x = Conv2D(64, (3, 3), padding='same', name='block1_conv2')(x)
  x = BatchNormalization()(x)
  block_1_out = Activation('relu')(x)
  
  x = MaxPooling2D()(block_1_out)
  
  # Block 2
  x = Conv2D(128, (3, 3), padding='same', name='block2_conv1')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  
  x = Conv2D(128, (3, 3), padding='same', name='block2_conv2')(x)
  x = BatchNormalization()(x)
  block_2_out = Activation('relu')(x)
  
  x = MaxPooling2D()(block_2_out)
  
  # Block 3
  x = Conv2D(256, (3, 3), padding='same', name='block3_conv1')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  
  x = Conv2D(256, (3, 3), padding='same', name='block3_conv2')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  
  x = Conv2D(256, (3, 3), padding='same', name='block3_conv3')(x)
  x = BatchNormalization()(x)
  block_3_out = Activation('relu')(x)
  
  x = MaxPooling2D()(block_3_out)
  
  # Block 4
  x = Conv2D(512, (3, 3), padding='same', name='block4_conv1')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  
  x = Conv2D(512, (3, 3), padding='same', name='block4_conv2')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  
  x = Conv2D(512, (3, 3), padding='same', name='block4_conv3')(x)
  x = BatchNormalization()(x)
  block_4_out = Activation('relu')(x)
  
  x = MaxPooling2D()(block_4_out)
  
  # Block 5
  x = Conv2D(512, (3, 3), padding='same', name='block5_conv1')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  
  x = Conv2D(512, (3, 3), padding='same', name='block5_conv2')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  
  x = Conv2D(512, (3, 3), padding='same', name='block5_conv3')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  
  for_pretrained_weight = MaxPooling2D()(x)
  
  # Load pretrained weights.
  if vgg_weight_path is not None:
    vgg16 = Model(img_input, for_pretrained_weight)
    vgg16.load_weights(vgg_weight_path, by_name=True)
  
  # UP 1
  x = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  
  block_4_out = att_modle(x, block_4_out, 512)
  x = concatenate([x, block_4_out])
  x = Conv2D(512, (3, 3), padding='same')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  
  x = Conv2D(512, (3, 3), padding='same')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  
  # UP 2
  x = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  
  block_3_out = att_modle(x, block_3_out, 256)
  x = concatenate([x, block_3_out])
  x = Conv2D(256, (3, 3), padding='same')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  
  x = Conv2D(256, (3, 3), padding='same')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  
  # UP 3
  x = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  
  block_2_out = att_modle(x, block_2_out, 128)
  x = concatenate([x, block_2_out])
  x = Conv2D(128, (3, 3), padding='same')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  
  x = Conv2D(128, (3, 3), padding='same')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  
  # UP 4
  x = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  
  block_1_out = att_modle(x, block_1_out, 64)
  x = concatenate([x, block_1_out])
  x = Conv2D(64, (3, 3), padding='same')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  
  x = Conv2D(64, (3, 3), padding='same')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  
  # last conv
  x = Conv2D(num_classes, (3, 3), activation='sigmoid', padding='same')(x)
  
  model = Model(img_input, x)
  
  return model


def R2unet(num_classes=1, input_shape=(224, 224, 1), vgg_weight_path=None):
  img_input = Input(input_shape)
  
  # Block 1
  x = Conv2D(64, (3, 3), padding='same', name='block1_conv1')(img_input)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  
  x = Conv2D(64, (3, 3), padding='same', name='block1_conv2')(x)
  x = BatchNormalization()(x)
  block_1_out = Activation('relu')(x)
  
  x = MaxPooling2D()(block_1_out)
  
  # Block 2
  x = Conv2D(128, (3, 3), padding='same', name='block2_conv1')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  
  x = Conv2D(128, (3, 3), padding='same', name='block2_conv2')(x)
  x = BatchNormalization()(x)
  block_2_out = Activation('relu')(x)
  
  x = MaxPooling2D()(block_2_out)
  
  # Block 3
  x = Conv2D(256, (3, 3), padding='same', name='block3_conv1')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  
  x = Conv2D(256, (3, 3), padding='same', name='block3_conv2')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  
  x = Conv2D(256, (3, 3), padding='same', name='block3_conv3')(x)
  x = BatchNormalization()(x)
  block_3_out = Activation('relu')(x)
  
  x = MaxPooling2D()(block_3_out)
  
  # Block 4
  x = Conv2D(512, (3, 3), padding='same', name='block4_conv1')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  
  x = Conv2D(512, (3, 3), padding='same', name='block4_conv2')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  
  x = Conv2D(512, (3, 3), padding='same', name='block4_conv3')(x)
  x = BatchNormalization()(x)
  block_4_out = Activation('relu')(x)
  
  x = MaxPooling2D()(block_4_out)
  
  # Block 5
  x = Conv2D(512, (3, 3), padding='same', name='block5_conv1')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  
  x = Conv2D(512, (3, 3), padding='same', name='block5_conv2')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  
  x = Conv2D(512, (3, 3), padding='same', name='block5_conv3')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  
  for_pretrained_weight = MaxPooling2D()(x)
  
  # Load pretrained weights.
  if vgg_weight_path is not None:
    vgg16 = Model(img_input, for_pretrained_weight)
    vgg16.load_weights(vgg_weight_path, by_name=True)
  
  # UP 1
  x = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  
  x = concatenate([x, block_4_out])
  x = RCNN_block(x, 512, 3)
  x = Conv2D(512, (3, 3), padding='same')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  
  x = Conv2D(512, (3, 3), padding='same')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  
  # UP 2
  x = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  
  x = concatenate([x, block_3_out])
  x = RCNN_block(x, 256, 3)
  x = Conv2D(256, (3, 3), padding='same')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  
  x = Conv2D(256, (3, 3), padding='same')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  
  # UP 3
  x = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  
  x = concatenate([x, block_2_out])
  x = RCNN_block(x, 128, 3)
  x = Conv2D(128, (3, 3), padding='same')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  
  x = Conv2D(128, (3, 3), padding='same')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  
  # UP 4
  x = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  
  x = concatenate([x, block_1_out])
  x = RCNN_block(x, 64, 3)
  x = Conv2D(64, (3, 3), padding='same')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  
  x = Conv2D(64, (3, 3), padding='same')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  
  # last conv
  x = Conv2D(num_classes, (3, 3), activation='sigmoid', padding='same')(x)
  
  model = Model(img_input, x)
  
  return model


def AttR2unet(num_classes=1, input_shape=(224, 224, 1), vgg_weight_path=None):
  img_input = Input(input_shape)
  
  # Block 1
  x = Conv2D(64, (3, 3), padding='same', name='block1_conv1')(img_input)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  
  x = Conv2D(64, (3, 3), padding='same', name='block1_conv2')(x)
  x = BatchNormalization()(x)
  block_1_out = Activation('relu')(x)
  
  x = MaxPooling2D()(block_1_out)
  
  # Block 2
  x = Conv2D(128, (3, 3), padding='same', name='block2_conv1')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  
  x = Conv2D(128, (3, 3), padding='same', name='block2_conv2')(x)
  x = BatchNormalization()(x)
  block_2_out = Activation('relu')(x)
  
  x = MaxPooling2D()(block_2_out)
  
  # Block 3
  x = Conv2D(256, (3, 3), padding='same', name='block3_conv1')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  
  x = Conv2D(256, (3, 3), padding='same', name='block3_conv2')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  
  x = Conv2D(256, (3, 3), padding='same', name='block3_conv3')(x)
  x = BatchNormalization()(x)
  block_3_out = Activation('relu')(x)
  
  x = MaxPooling2D()(block_3_out)
  
  # Block 4
  x = Conv2D(512, (3, 3), padding='same', name='block4_conv1')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  
  x = Conv2D(512, (3, 3), padding='same', name='block4_conv2')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  
  x = Conv2D(512, (3, 3), padding='same', name='block4_conv3')(x)
  x = BatchNormalization()(x)
  block_4_out = Activation('relu')(x)
  
  x = MaxPooling2D()(block_4_out)
  
  # Block 5
  x = Conv2D(512, (3, 3), padding='same', name='block5_conv1')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  
  x = Conv2D(512, (3, 3), padding='same', name='block5_conv2')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  
  x = Conv2D(512, (3, 3), padding='same', name='block5_conv3')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  
  for_pretrained_weight = MaxPooling2D()(x)
  
  # Load pretrained weights.
  if vgg_weight_path is not None:
    vgg16 = Model(img_input, for_pretrained_weight)
    vgg16.load_weights(vgg_weight_path, by_name=True)
  
  # UP 1
  x = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  
  block_4_out = att_modle(x, block_4_out, 512)
  x = concatenate([x, block_4_out])
  x = RCNN_block(x, 512, 3)
  
  x = Conv2D(512, (3, 3), padding='same')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  
  x = Conv2D(512, (3, 3), padding='same')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  
  # UP 2
  x = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  
  block_3_out = att_modle(x, block_3_out, 256)
  x = concatenate([x, block_3_out])
  x = RCNN_block(x, 256, 3)
  
  x = Conv2D(256, (3, 3), padding='same')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  
  x = Conv2D(256, (3, 3), padding='same')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  
  # UP 3
  x = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  
  block_2_out = att_modle(x, block_2_out, 128)
  x = concatenate([x, block_2_out])
  x = RCNN_block(x, 128, 3)
  
  x = Conv2D(128, (3, 3), padding='same')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  
  x = Conv2D(128, (3, 3), padding='same')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  
  # UP 4
  x = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  
  block_1_out = att_modle(x, block_1_out, 64)
  x = concatenate([x, block_1_out])
  x = RCNN_block(x, 64, 3)
  
  x = Conv2D(64, (3, 3), padding='same')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  
  x = Conv2D(64, (3, 3), padding='same')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  
  # last conv
  x = Conv2D(num_classes, (3, 3), activation='sigmoid', padding='same')(x)
  
  model = Model(img_input, x)
  
  return model


def identity_block(input_tensor, kernel_size, filters, stage, block):
  """The identity block is the block that has no conv layer at shortcut.

  # Arguments
      input_tensor: input tensor
      kernel_size: default 3, the kernel size of
          middle conv layer at main path
      filters: list of integers, the filters of 3 conv layer at main path
      stage: integer, current stage label, used for generating layer names
      block: 'a','b'..., current block label, used for generating layer names

  # Returns
      Output tensor for the block.
  """
  filters1, filters2, filters3 = filters
  if K.image_data_format() == 'channels_last':
    bn_axis = 3
  else:
    bn_axis = 1
  conv_name_base = 'res' + str(stage) + block + '_branch'
  bn_name_base = 'bn' + str(stage) + block + '_branch'
  
  x = Conv2D(filters1, (1, 1),
             name=conv_name_base + '2a')(input_tensor)
  x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
  x = Activation('relu')(x)
  
  x = Conv2D(filters2, kernel_size,
             padding='same', name=conv_name_base + '2b')(x)
  x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
  x = Activation('relu')(x)
  
  x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
  x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)
  
  x = add([x, input_tensor])
  x = Activation('relu')(x)
  return x


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
  """A block that has a conv layer at shortcut.

  # Arguments
      input_tensor: input tensor
      kernel_size: default 3, the kernel size of
          middle conv layer at main path
      filters: list of integers, the filters of 3 conv layer at main path
      stage: integer, current stage label, used for generating layer names
      block: 'a','b'..., current block label, used for generating layer names
      strides: Strides for the first conv layer in the block.

  # Returns
      Output tensor for the block.

  Note that from stage 3,
  the first conv layer at main path is with strides=(2, 2)
  And the shortcut should have strides=(2, 2) as well
  """
  filters1, filters2, filters3 = filters
  if K.image_data_format() == 'channels_last':
    bn_axis = 3
  else:
    bn_axis = 1
  conv_name_base = 'res' + str(stage) + block + '_branch'
  bn_name_base = 'bn' + str(stage) + block + '_branch'
  
  x = Conv2D(filters1, (1, 1), strides=strides,
             name=conv_name_base + '2a')(input_tensor)
  x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
  x = Activation('relu')(x)
  
  x = Conv2D(filters2, kernel_size, padding='same',
             name=conv_name_base + '2b')(x)
  x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
  x = Activation('relu')(x)
  
  x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
  x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)
  
  shortcut = Conv2D(filters3, (1, 1), strides=strides,
                    name=conv_name_base + '1')(input_tensor)
  shortcut = BatchNormalization(
    axis=bn_axis, name=bn_name_base + '1')(shortcut)
  
  x = add([x, shortcut])
  x = Activation('relu')(x)
  return x


def ResNetunet(num_classes=1, input_shape=(224, 224, 6)):
  img_input = Input(input_shape)
  
  x = Conv2D(32, (3, 3), padding='same', name='block1_conv1')(img_input)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  
  x = Conv2D(64, (3, 3), padding='same', name='block1_conv2')(x)
  x = BatchNormalization()(x)
  
  block_1_out = Activation('relu')(x)
  
  x = MaxPooling2D((2, 2), strides=(2, 2))(block_1_out)
  
  x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
  x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
  block_2_out = identity_block(x, 3, [64, 64, 256], stage=2, block='c')
  
  x = conv_block(block_2_out, 3, [128, 128, 512], stage=3, block='a')
  x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
  x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
  block_3_out = identity_block(x, 3, [128, 128, 512], stage=3, block='d')
  
  x = conv_block(block_3_out, 3, [256, 256, 1024], stage=4, block='a')
  x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
  x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
  x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
  x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
  block_4_out = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')
  
  x = conv_block(block_4_out, 3, [512, 512, 2048], stage=5, block='a')
  x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
  x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')
  
  # UP 1
  x = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  
  x = concatenate([x, block_4_out])
  x = Conv2D(512, (3, 3), padding='same')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  
  x = Conv2D(512, (3, 3), padding='same')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  
  # UP 2
  x = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  
  x = concatenate([x, block_3_out])
  x = Conv2D(256, (3, 3), padding='same')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  
  x = Conv2D(256, (3, 3), padding='same')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  
  # UP 3
  x = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  
  x = concatenate([x, block_2_out])
  x = Conv2D(128, (3, 3), padding='same')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  
  x = Conv2D(128, (3, 3), padding='same')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  
  # UP 4
  x = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  
  x = concatenate([x, block_1_out])
  x = Conv2D(64, (3, 3), padding='same')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  
  x = Conv2D(64, (3, 3), padding='same')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  
  # last conv
  x = Conv2D(num_classes, (3, 3), activation='sigmoid', padding='same')(x)
  
  model = Model(img_input, x)
  
  return model


model = unet(1,(101,101,1))
model.summary()