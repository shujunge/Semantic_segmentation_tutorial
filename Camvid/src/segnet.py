
from keras.layers import *
from keras.models import *
from keras.callbacks import *
import keras.backend as K



class UnPooling2D(Layer):
  """A 2D Repeat layer"""
  
  def __init__(self, poolsize=(2, 2)):
    super(UnPooling2D, self).__init__()
    self.poolsize = poolsize
  
  @property
  def output_shape(self):
    input_shape = self.input_shape
    return (input_shape[0], input_shape[1],
            self.poolsize[0] * input_shape[2],
            self.poolsize[1] * input_shape[3])
  
  def get_output(self, train):
    X = self.get_input(train)
    s1 = self.poolsize[0]
    s2 = self.poolsize[1]
    output = X.repeat(s1, axis=2).repeat(s2, axis=3)
    return output
  
  def get_config(self):
    return {"name": self.__class__.__name__,
            "poolsize": self.poolsize}


def create_encoding_layers():
  kernel = 3
  filter_size = 64
  pad = 1
  pool_size = 2
  return [
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


def create_decoding_layers():
  kernel = 3
  filter_size = 64
  pad = 1
  pool_size = 2
  return [
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


def segnet(nclasses=12, input_shape=(224,224, 3)):
  autoencoder = Sequential()
  autoencoder.add(Layer(input_shape=input_shape))
  autoencoder.encoding_layers = create_encoding_layers()
  autoencoder.decoding_layers = create_decoding_layers()
  for l in autoencoder.encoding_layers:
    autoencoder.add(l)
  for l in autoencoder.decoding_layers:
    autoencoder.add(l)
  
  autoencoder.add(Conv2D(nclasses, (1, 1), padding='valid'))
  autoencoder.add(Activation('softmax'))
  return autoencoder