
from keras.layers import *
from keras.models import *
from keras.callbacks import *
import keras.backend as K

def SegNet_UnSamping(nclasses=12, input_shape=(224,224, 3)):
  kernel = 3
  filter_size = 64
  pad = 1
  pool_size = 2
  
  autoencoder = Sequential()
  autoencoder.add(Layer(input_shape=input_shape))

  autoencoder.encoding_layers = [
    ZeroPadding2D(padding=(pad, pad)),
    Conv2D(filter_size, (kernel, kernel), padding='valid'),
    BatchNormalization(),
    Activation('relu'),
    MaxPooling2D(pool_size=(pool_size, pool_size)),
    
    ZeroPadding2D(padding=(pad, pad)),
    Conv2D(128, (kernel, kernel), padding='valid'),
    BatchNormalization(),
    Activation('relu'),
    MaxPooling2D(pool_size=(pool_size, pool_size)),
    
    ZeroPadding2D(padding=(pad, pad)),
    Conv2D(256, (kernel, kernel), padding='valid'),
    BatchNormalization(),
    Activation('relu'),
    MaxPooling2D(pool_size=(pool_size, pool_size)),
    
    ZeroPadding2D(padding=(pad, pad)),
    Convolution2D(512, (kernel, kernel), padding='valid'),
    BatchNormalization(),
    Activation('relu'),
  
  ]
  
  autoencoder.decoding_layers = [
    ZeroPadding2D(padding=(pad, pad)),
    Conv2D(512, (kernel, kernel), padding='valid'),
    BatchNormalization(),
    
    UpSampling2D(size=(pool_size, pool_size)),
    ZeroPadding2D(padding=(pad, pad)),
    Conv2D(256, (kernel, kernel), padding='valid'),
    BatchNormalization(),
    
    UpSampling2D(size=(pool_size, pool_size)),
    ZeroPadding2D(padding=(pad, pad)),
    Conv2D(128, (kernel, kernel), padding='valid'),
    BatchNormalization(),
    
    UpSampling2D(size=(pool_size, pool_size)),
    ZeroPadding2D(padding=(pad, pad)),
    Conv2D(filter_size, (kernel, kernel), padding='valid'),
    BatchNormalization(),
  ]
  
  for l in autoencoder.encoding_layers:
    autoencoder.add(l)
  for l in autoencoder.decoding_layers:
    autoencoder.add(l)
  
  autoencoder.add(Conv2D(nclasses, (1, 1), padding='valid'))
  autoencoder.add(Activation('softmax'))
  
  return autoencoder


def SegNet_DeConv(nclasses=12, input_shape=(224,224, 3)):
  kernel = 3
  filter_size = 64
  pad = 1
  pool_size = 2
  
  autoencoder = Sequential()
  autoencoder.add(Layer(input_shape=input_shape))
  autoencoder.encoding_layers = [
    ZeroPadding2D(padding=(pad, pad)),
    Conv2D(filter_size, (kernel, kernel), padding='valid'),
    BatchNormalization(),
    Activation('relu'),
    MaxPooling2D(pool_size=(pool_size, pool_size)),
    
    ZeroPadding2D(padding=(pad, pad)),
    Conv2D(128, (kernel, kernel), padding='valid'),
    BatchNormalization(),
    Activation('relu'),
    MaxPooling2D(pool_size=(pool_size, pool_size)),
    
    ZeroPadding2D(padding=(pad, pad)),
    Conv2D(256, (kernel, kernel), padding='valid'),
    BatchNormalization(),
    Activation('relu'),
    MaxPooling2D(pool_size=(pool_size, pool_size)),
    
    ZeroPadding2D(padding=(pad, pad)),
    Convolution2D(512, (kernel, kernel), padding='valid'),
    BatchNormalization(),
    Activation('relu'),
  
  ]
  
  autoencoder.decoding_layers = [ZeroPadding2D(padding=(pad, pad)),
    Conv2D(512, (kernel, kernel), padding='valid'),
    BatchNormalization(),
  
    Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same'),
    ZeroPadding2D(padding=(pad, pad)),
    Conv2D(256, (kernel, kernel), padding='valid'),
    BatchNormalization(),
  
    Conv2DTranspose(256,(2, 2), strides=(2, 2), padding='same'),
    ZeroPadding2D(padding=(pad, pad)),
    Conv2D(128, (kernel, kernel), padding='valid'),
    BatchNormalization(),
  
    Conv2DTranspose(128,(2, 2), strides=(2, 2), padding='same'),
    ZeroPadding2D(padding=(pad, pad)),
    Conv2D(filter_size, (kernel, kernel), padding='valid'),
    BatchNormalization(),
  ]

  for l in autoencoder.encoding_layers:
    autoencoder.add(l)
  for l in autoencoder.decoding_layers:
    autoencoder.add(l)
  
  autoencoder.add(Conv2D(nclasses, (1, 1), padding='valid'))
  autoencoder.add(Activation('softmax'))
  return autoencoder


class MaxPoolingWithArgmax2D(Layer):

  def __init__(
    self,
    pool_size=(2, 2),
    strides=(2, 2),
    padding='same',
    **kwargs):
    super(MaxPoolingWithArgmax2D, self).__init__(**kwargs)
    self.padding = padding
    self.pool_size = pool_size
    self.strides = strides

  def call(self, inputs, **kwargs):
    padding = self.padding
    pool_size = self.pool_size
    strides = self.strides
    if K.backend() == 'tensorflow':
      ksize = [1, pool_size[0], pool_size[1], 1]
      padding = padding.upper()
      strides = [1, strides[0], strides[1], 1]
      output, argmax = K.tf.nn.max_pool_with_argmax(
        inputs,
        ksize=ksize,
        strides=strides,
        padding=padding)
    else:
      errmsg = '{} backend is not supported for layer {}'.format(
        K.backend(), type(self).__name__)
      raise NotImplementedError(errmsg)
    argmax = K.cast(argmax, K.floatx())
    return [output, argmax]

  def compute_output_shape(self, input_shape):
    ratio = (1, 2, 2, 1)
    output_shape = [
      dim // ratio[idx]
      if dim is not None else None
      for idx, dim in enumerate(input_shape)]
    output_shape = tuple(output_shape)
    return [output_shape, output_shape]

  def compute_mask(self, inputs, mask=None):
    return 2 * [None]

class MaxUnpooling2D(Layer):
  def __init__(self, size=(2, 2), **kwargs):
    super(MaxUnpooling2D, self).__init__(**kwargs)
    self.size = size

  def call(self, inputs, output_shape=None):
    updates, mask = inputs[0], inputs[1]
    with K.tf.variable_scope(self.name):
      mask = K.cast(mask, 'int32')
      input_shape = K.tf.shape(updates, out_type='int32')
      #  calculation new shape
      if output_shape is None:
        output_shape = (
          input_shape[0],
          input_shape[1] * self.size[0],
          input_shape[2] * self.size[1],
          input_shape[3])
      self.output_shape1 = output_shape
    
      # calculation indices for batch, height, width and feature maps
      one_like_mask = K.ones_like(mask, dtype='int32')
      batch_shape = K.concatenate(
        [[input_shape[0]], [1], [1], [1]],
        axis=0)
      batch_range = K.reshape(
        K.tf.range(output_shape[0], dtype='int32'),
        shape=batch_shape)
      b = one_like_mask * batch_range
      y = mask // (output_shape[2] * output_shape[3])
      x = (mask // output_shape[3]) % output_shape[2]
      feature_range = K.tf.range(output_shape[3], dtype='int32')
      f = one_like_mask * feature_range
    
      # transpose indices & reshape update values to one dimension
      updates_size = K.tf.size(updates)
      indices = K.transpose(K.reshape(
        K.stack([b, y, x, f]),
        [4, updates_size]))
      values = K.reshape(updates, [updates_size])
      ret = K.tf.scatter_nd(indices, values, output_shape)
      return ret

  def compute_output_shape(self, input_shape):
    mask_shape = input_shape[1]
    return (
      mask_shape[0],
      mask_shape[1] * self.size[0],
      mask_shape[2] * self.size[1],
      mask_shape[3]
    )

def SegNet_Unpooling(
  input_shape=(360,480, 3),
  nclasses=12,
  kernel=3,filter_size=64,
  pool_size=2,pad=1,
  output_mode="softmax"):
  # encoder
  inputs = Input(shape=input_shape)

  conv1 = ZeroPadding2D(padding=(pad, pad))(inputs)
  conv1 = Conv2D(filter_size, (kernel, kernel), padding='valid')(conv1)
  conv1 = BatchNormalization()(conv1)
  conv1 = Activation('relu')(conv1)

  conv1, mask_1 = MaxPoolingWithArgmax2D(pool_size=(pool_size, pool_size))(conv1)

  conv2 = ZeroPadding2D(padding=(pad, pad))(conv1)
  conv2 =Conv2D(128, (kernel, kernel), padding='valid')(conv2)
  conv2 = BatchNormalization()(conv2)
  conv2 = Activation('relu')(conv2)
  
  conv2, mask_2 = MaxPoolingWithArgmax2D(pool_size=(pool_size, pool_size))(conv2)

  conv3 = ZeroPadding2D(padding=(pad, pad))(conv2)
  conv3 = Conv2D(256, (kernel, kernel), padding='valid')(conv3)
  conv3 = BatchNormalization()(conv3)
  conv3 = Activation('relu')(conv3)
  
  conv3, mask_3 = MaxPoolingWithArgmax2D(pool_size=(pool_size, pool_size))(conv3)

  conv4 = ZeroPadding2D(padding=(pad, pad))(conv3)
  conv4 = Convolution2D(512, (kernel, kernel), padding='valid')(conv4)
  conv4 = BatchNormalization()(conv4)
  conv4 = Activation('relu')(conv4)
  # decoder

  up1 = ZeroPadding2D(padding=(pad, pad))(conv4)
  up1 = Conv2D(256, (kernel, kernel), padding='valid')(up1)
  up1 = BatchNormalization()(up1)

  up2 = MaxUnpooling2D((pool_size, pool_size))([up1, mask_3])
  up2 = ZeroPadding2D(padding=(pad, pad))(up2)
  up2 = Conv2D(128, (kernel, kernel), padding='valid')(up2)
  up2 = BatchNormalization()(up2)

  up3 = MaxUnpooling2D((pool_size, pool_size))([up2, mask_2])
  up3 = ZeroPadding2D(padding=(pad, pad))(up3)
  up3 = Conv2D(64, (kernel, kernel), padding='valid')(up3)
  up3 = BatchNormalization()(up3)

  up4 = MaxUnpooling2D((pool_size, pool_size))([up3, mask_1])
  up4 = ZeroPadding2D(padding=(pad, pad))(up4)
  up4 = Conv2D(filter_size, (kernel, kernel), padding='valid')(up4)
  up4 = BatchNormalization()(up4)

  conv_26 = Convolution2D(nclasses, (1, 1), padding="valid")(up4)
  outputs = Activation(output_mode)(conv_26)
  print("Build decoder done..")

  model = Model(inputs=inputs, outputs=outputs, name="SegNet")

  return model


# model =SegNet_Unpooling()
# model.summary()