
import keras.backend as K

import tensorflow as tf
def focal_loss(y_true, y_pred):
  eps = 1e-12
  gamma = 2
  alpha = 0.75
  y_pred=K.clip(y_pred,eps,1.-eps)
  pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
  pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
  return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1))-K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0))


def binary_crossentropy(y_true, y_pred):
  return K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1)


def dice_coef_loss(y_true, y_pred):
  def dice_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(y_true * y_pred, axis=[1, 2, 3])
    union = K.sum(y_true, axis=[1, 2, 3]) + K.sum(y_pred, axis=[1, 2, 3])
    return K.mean((2. * intersection + smooth) / (union + smooth), axis=0)
  return 1 - dice_coef(y_true, y_pred, smooth=1)

def IoU(y_true, y_pred, eps=1e-6):
    if K.max(y_true) == 0.0:
        return IoU(1-y_true, 1-y_pred) ## empty image; calc IoU of zeros
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3]) - intersection
    return K.mean( (intersection + eps) / (union + eps), axis=0)



def tversky_loss(y_true, y_pred):
  def tversky(y_true, y_pred, smooth=1):
    y_true_pos = K.flatten(y_true)
    y_pred_pos = K.flatten(y_pred)
    true_pos = K.sum(y_true_pos * y_pred_pos)
    false_neg = K.sum(y_true_pos * (1 - y_pred_pos))
    false_pos = K.sum((1 - y_true_pos) * y_pred_pos)
    alpha = 0.7
    return (true_pos + smooth) / (true_pos + alpha * false_neg + (1 - alpha) * false_pos + smooth)
  return 1 - tversky(y_true, y_pred)


def focal_tversky(y_true, y_pred):
  def tversky(y_true, y_pred, smooth=1):
    y_true_pos = K.flatten(y_true)
    y_pred_pos = K.flatten(y_pred)
    true_pos = K.sum(y_true_pos * y_pred_pos)
    false_neg = K.sum(y_true_pos * (1 - y_pred_pos))
    false_pos = K.sum((1 - y_true_pos) * y_pred_pos)
    alpha = 0.7
    return (true_pos + smooth) / (true_pos + alpha * false_neg + (1 - alpha) * false_pos + smooth)
  pt_1 = tversky(y_true, y_pred)
  gamma = 0.75
  return K.pow((1 - pt_1), gamma)

def matthews_correlation(y_true, y_pred):
  '''Calculates the Matthews correlation coefficient measure for quality
  of binary classification problems.
  '''
  y_pred_pos = K.round(K.clip(y_pred, 0, 1))
  y_pred_neg = 1 - y_pred_pos

  y_pos = K.round(K.clip(y_true, 0, 1))
  y_neg = 1 - y_pos

  tp = K.sum(y_pos * y_pred_pos)
  tn = K.sum(y_neg * y_pred_neg)

  fp = K.sum(y_neg * y_pred_pos)
  fn = K.sum(y_pos * y_pred_neg)

  numerator = (tp * tn - fp * fn)
  denominator = K.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

  return numerator / (denominator + K.epsilon())