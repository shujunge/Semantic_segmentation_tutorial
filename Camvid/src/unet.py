from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, Activation
from keras.layers import BatchNormalization
from keras import backend as K


def dice_coef(y_true, y_pred):
    return (2. * K.sum(y_true * y_pred) + 1.) / (K.sum(y_true) + K.sum(y_pred) + 1.)


def unet(num_classes=1, input_shape=(224,224,1),vgg_weight_path=None):
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
    x = Conv2D(1024, (3, 3), padding='same', name='block5_conv1')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(1024, (3, 3), padding='same', name='block5_conv2')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(1024, (3, 3), padding='same', name='block5_conv3')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    for_pretrained_weight = MaxPooling2D()(x)

    # Load pretrained weights.
    if vgg_weight_path is not None:
        vgg16 = Model(img_input, for_pretrained_weight)
        vgg16.load_weights(vgg_weight_path, by_name=True)

    # UP 1
    x = Conv2DTranspose(1024, (2, 2), strides=(2, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = concatenate([x, block_4_out])
    x = Conv2D(1024, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(1024, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # UP 2
    x = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = concatenate([x, block_3_out])
    x = Conv2D(512, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(512, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # UP 3
    x = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = concatenate([x, block_2_out])
    x = Conv2D(256, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(256, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # UP 4
    x = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = concatenate([x, block_1_out])
    x = Conv2D(128, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(128, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # last conv
    x = Conv2D(num_classes, (3, 3), activation='softmax', padding='same')(x)

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
  x = Conv2D(num_classes, (3, 3), activation='softmax', padding='same')(x)
  
  model = Model(img_input, x)
  
  return model





def unetplus(num_classes=1, input_shape=(224, 224, 1)):
  
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

  block_2_up = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(block_2_out)
  block_2_up = BatchNormalization()(block_2_up)
  block_2_up = Activation('relu')(block_2_up)
  temp_2 = concatenate([block_2_up, block_1_out])
  down2_up = Conv2D(32, (3, 3), padding='same')(temp_2)
  down2_up = BatchNormalization()(down2_up)
  down2_up = Activation('relu')(down2_up)
  down2_out = Conv2D(1, (3, 3), padding='same',activation='sigmoid')(down2_up)
  
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
  
  block_3_up1 = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(block_3_out)
  block_3_up1 = BatchNormalization()(block_3_up1)
  block_3_up1 = Activation('relu')(block_3_up1)
  temp_31 = concatenate([block_3_up1, block_2_out])
  block_3_up1 = Conv2D(32, (3, 3), padding='same')(temp_31)
  block_3_up1 = BatchNormalization()(block_3_up1)
  block_3_up1 = Activation('relu')(block_3_up1)
  
  block_3_up2 = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(block_3_up1)
  block_3_up2 = BatchNormalization()(block_3_up2)
  block_3_up2 = Activation('relu')(block_3_up2)

  temp_32 = concatenate([block_3_up2, block_1_out,down2_up])
  block_3_up2 = Conv2D(32, (3, 3), padding='same')(temp_32)
  block_3_up2 = BatchNormalization()(block_3_up2)
  block_3_up2 = Activation('relu')(block_3_up2)
  
  down3_out = Conv2D(1, (3, 3), padding='same',activation='sigmoid')(block_3_up2)

  
  # Block 4
  x = Conv2D(512, (3, 3), padding='same', name='block4_conv1')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  
  x = Conv2D(512, (3, 3), padding='same', name='block4_conv2')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  
  x = Conv2D(512, (3, 3), padding='same', name='block4_conv3')(x)
  x = BatchNormalization()(x)
  block_4_out= Activation('relu')(x)

  x = MaxPooling2D()(block_4_out)
  
  block_4_up1 = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(block_4_out)
  block_4_up1 = BatchNormalization()(block_4_up1)
  block_4_up1 = Activation('relu')(block_4_up1)
  temp_41 = concatenate([block_4_up1, block_3_out])
  block_4_up1 = Conv2D(32, (3, 3), padding='same')(temp_41)
  block_4_up1 = BatchNormalization()(block_4_up1)
  block_4_up1 = Activation('relu')(block_4_up1)
  

  block_4_up2 = Conv2DTranspose(64, (3, 3), strides=(2,2), padding='same')(block_4_up1)
  block_4_up2 = BatchNormalization()(block_4_up2)
  block_4_up2 = Activation('relu')(block_4_up2)
  temp_42 = concatenate([block_4_up2, block_2_out,block_3_up1])
  block_4_up2 = Conv2D(32, (3, 3), padding='same')(temp_42)
  block_4_up2 = BatchNormalization()(block_4_up2)
  block_4_up2 = Activation('relu')(block_4_up2)
  
  block_4_up3 = Conv2DTranspose(64, (3, 3), strides=(2,2), padding='same')(block_4_up2)
  block_4_up3 = BatchNormalization()(block_4_up3)
  block_4_up3 = Activation('relu')(block_4_up3)
  temp_43 = concatenate([block_4_up3, block_1_out,block_3_up2,block_2_up])
  block_4_up3 = Conv2D(32, (3, 3), padding='same')(temp_43)
  block_4_up3 = BatchNormalization()(block_4_up3)
  block_4_up3 = Activation('relu')(block_4_up3)

  down4_out = Conv2D(1, (3, 3), padding='same',activation='sigmoid')(block_4_up3)
  
  # Block 5
  x = Conv2D(512, (3, 3), padding='same', name='block5_conv1')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  
  x = Conv2D(512, (3, 3), padding='same', name='block5_conv2')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  
  x = Conv2D(512, (3, 3), padding='same', name='block5_conv3')(x)
  x = BatchNormalization()(x)
  block_5_out = Activation('relu')(x)
  
  
  
  # UP 1
  x = Conv2DTranspose(512, (3, 3), strides=(2, 2), padding='same')(block_5_out)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  
  x = concatenate([x, block_4_out])
  x = Conv2D(512, (3, 3), padding='same')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  
  x = Conv2D(512, (3, 3), padding='same')(x)
  x = BatchNormalization()(x)
  UP_1 = Activation('relu')(x)
  
  # UP 2
  x = Conv2DTranspose(256, (3, 3), strides=(2, 2), padding='same')(UP_1)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)

  block_3_out = concatenate([block_3_out,block_4_up1])
  x = concatenate([x, block_3_out])
  x = Conv2D(256, (3, 3), padding='same')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  
  x = Conv2D(256, (3, 3), padding='same')(x)
  x = BatchNormalization()(x)
  UP_2 = Activation('relu')(x)
  
  # UP 3
  x = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(UP_2)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)

  x = concatenate([x, block_2_out, block_3_up1, block_4_up2])
  x = Conv2D(128, (3, 3), padding='same')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  
  x = Conv2D(128, (3, 3), padding='same')(x)
  x = BatchNormalization()(x)
  UP_3 = Activation('relu')(x)
  
  # UP 4
  x = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(UP_3)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
 
  
  x = concatenate([x, block_1_out, block_2_up, block_3_up2, block_4_up3])
  x = Conv2D(64, (3, 3), padding='same')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  
  x = Conv2D(64, (3, 3), padding='same')(x)
  x = BatchNormalization()(x)
  UP_4 = Activation('relu')(x)
  
  # last conv
  x = Conv2D(num_classes, (3, 3), activation='sigmoid', padding='same')(UP_4)
  
  model = Model(img_input, [down2_out,down3_out,down4_out,x])
  
  return model

def fiveunetplus(num_classes=1, input_shape=(224, 224, 1)):
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
  
  block_2_up = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(block_2_out)
  block_2_up = BatchNormalization()(block_2_up)
  block_2_up = Activation('relu')(block_2_up)
  temp_2 = concatenate([block_2_up, block_1_out])
  down2_up = Conv2D(32, (3, 3), padding='same')(temp_2)
  down2_up = BatchNormalization()(down2_up)
  down2_up = Activation('relu')(down2_up)
  down2_out = Conv2D(1, (3, 3), padding='same', activation='sigmoid')(down2_up)
  
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
  
  block_3_up1 = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(block_3_out)
  block_3_up1 = BatchNormalization()(block_3_up1)
  block_3_up1 = Activation('relu')(block_3_up1)
  temp_31 = concatenate([block_3_up1, block_2_out])
  block_3_up1 = Conv2D(32, (3, 3), padding='same')(temp_31)
  block_3_up1 = BatchNormalization()(block_3_up1)
  block_3_up1 = Activation('relu')(block_3_up1)
  
  block_3_up2 = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(block_3_up1)
  block_3_up2 = BatchNormalization()(block_3_up2)
  block_3_up2 = Activation('relu')(block_3_up2)
  
  temp_32 = concatenate([block_3_up2, block_1_out, down2_up])
  block_3_up2 = Conv2D(32, (3, 3), padding='same')(temp_32)
  block_3_up2 = BatchNormalization()(block_3_up2)
  block_3_up2 = Activation('relu')(block_3_up2)
  
  down3_out = Conv2D(1, (3, 3), padding='same', activation='sigmoid')(block_3_up2)
  
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
  
  block_4_up1 = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(block_4_out)
  block_4_up1 = BatchNormalization()(block_4_up1)
  block_4_up1 = Activation('relu')(block_4_up1)
  temp_41 = concatenate([block_4_up1, block_3_out])
  block_4_up1 = Conv2D(32, (3, 3), padding='same')(temp_41)
  block_4_up1 = BatchNormalization()(block_4_up1)
  block_4_up1 = Activation('relu')(block_4_up1)
  
  block_4_up2 = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(block_4_up1)
  block_4_up2 = BatchNormalization()(block_4_up2)
  block_4_up2 = Activation('relu')(block_4_up2)
  temp_42 = concatenate([block_4_up2, block_2_out, block_3_up1])
  block_4_up2 = Conv2D(32, (3, 3), padding='same')(temp_42)
  block_4_up2 = BatchNormalization()(block_4_up2)
  block_4_up2 = Activation('relu')(block_4_up2)
  
  block_4_up3 = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(block_4_up2)
  block_4_up3 = BatchNormalization()(block_4_up3)
  block_4_up3 = Activation('relu')(block_4_up3)
  temp_43 = concatenate([block_4_up3, block_1_out, block_3_up2,block_2_up])
  block_4_up3 = Conv2D(32, (3, 3), padding='same')(temp_43)
  block_4_up3 = BatchNormalization()(block_4_up3)
  block_4_up3 = Activation('relu')(block_4_up3)
  
  down4_out = Conv2D(1, (3, 3), padding='same', activation='sigmoid')(block_4_up3)
  
  # Block 5
  x = Conv2D(512, (3, 3), padding='same', name='block5_conv1')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  
  x = Conv2D(512, (3, 3), padding='same', name='block5_conv2')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  
  x = Conv2D(512, (3, 3), padding='same', name='block5_conv3')(x)
  x = BatchNormalization()(x)
  
  block_5_out = Activation('relu')(x)
  x = MaxPooling2D()(block_5_out)

  block_5_up1 = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(block_5_out)
  block_5_up1 = BatchNormalization()(block_5_up1)
  block_5_up1 = Activation('relu')(block_5_up1)
  temp_51 = concatenate([block_5_up1, block_4_out])
  block_5_up1 = Conv2D(32, (3, 3), padding='same')(temp_51)
  block_5_up1 = BatchNormalization()(block_5_up1)
  block_5_up1 = Activation('relu')(block_5_up1)

  block_5_up2 = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(block_5_up1)
  block_5_up2 = BatchNormalization()(block_5_up2)
  block_5_up2 = Activation('relu')(block_5_up2)
  temp_52 = concatenate([block_5_up2, block_3_out, block_4_up1])
  block_5_up2 = Conv2D(32, (3, 3), padding='same')(temp_52)
  block_5_up2 = BatchNormalization()(block_5_up2)
  block_5_up2 = Activation('relu')(block_5_up2)

  block_5_up3 = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(block_5_up2)
  block_5_up3 = BatchNormalization()(block_5_up3)
  block_5_up3 = Activation('relu')(block_5_up3)
  temp_53 = concatenate([block_5_up3, block_2_out, block_4_up2])
  block_5_up3 = Conv2D(32, (3, 3), padding='same')(temp_53)
  block_5_up3 = BatchNormalization()(block_5_up3)
  block_5_up3 = Activation('relu')(block_5_up3)

  block_5_up4 = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(block_5_up3)
  block_5_up4 = BatchNormalization()(block_5_up4)
  block_5_up4 = Activation('relu')(block_5_up4)
  temp_54 = concatenate([block_5_up4, block_1_out, block_4_up3,block_3_up2,block_2_up])
  block_5_up4 = Conv2D(32, (3, 3), padding='same')(temp_54)
  block_5_up4 = BatchNormalization()(block_5_up4)
  block_5_up4 = Activation('relu')(block_5_up4)

  down5_out = Conv2D(1, (3, 3), padding='same', activation='sigmoid')(block_5_up4)
  

  # Block 6
  x = Conv2D(512, (3, 3), padding='same', name='block6_conv1')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)

  x = Conv2D(512, (3, 3), padding='same', name='block6_conv2')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)

  x = Conv2D(512, (3, 3), padding='same', name='block6_conv3')(x)
  block_6_out = BatchNormalization()(x)
  x = Activation('relu')(block_6_out)
  
  # UP 1
  x = Conv2DTranspose(512, (3, 3), strides=(2, 2), padding='same')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)

  x = concatenate([x, block_5_out])
  x = Conv2D(512, (3, 3), padding='same')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  
  x = Conv2D(512, (3, 3), padding='same')(x)
  x = BatchNormalization()(x)
  UP_1 = Activation('relu')(x)
  
  # UP 2
  x = Conv2DTranspose(256, (3, 3), strides=(2, 2), padding='same')(UP_1)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  
  x = concatenate([x, block_4_out, block_5_up1])
  x = Conv2D(256, (3, 3), padding='same')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  
  x = Conv2D(256, (3, 3), padding='same')(x)
  x = BatchNormalization()(x)
  UP_2 = Activation('relu')(x)
  
  # UP 3
  x = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(UP_2)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  
  x = concatenate([x, block_3_out, block_4_up1, block_5_up2])
  x = Conv2D(128, (3, 3), padding='same')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  
  x = Conv2D(128, (3, 3), padding='same')(x)
  x = BatchNormalization()(x)
  UP_3 = Activation('relu')(x)
  
  # UP 4
  x = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(UP_3)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  
  x = concatenate([x, block_2_out, block_3_up1, block_4_up2, block_5_up3])
  x = Conv2D(64, (3, 3), padding='same')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  
  x = Conv2D(64, (3, 3), padding='same')(x)
  x = BatchNormalization()(x)
  UP_4 = Activation('relu')(x)

  # UP 5
  x = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(UP_4)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)

  x = concatenate([x, block_1_out, block_2_up, block_3_up2, block_4_up3,block_5_up4])
  x = Conv2D(64, (3, 3), padding='same')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)

  x = Conv2D(64, (3, 3), padding='same')(x)
  x = BatchNormalization()(x)
  UP_5 = Activation('relu')(x)
  
  # last conv
  x = Conv2D(num_classes, (3, 3), activation='sigmoid', padding='same')(UP_5)
  
  model = Model(img_input, [down2_out, down3_out, down4_out, down5_out,x])
  
  return model

