import numpy as np
from tqdm import  tqdm

from VOC2012 import *

voc2012 = VOC2012('../../../VOCdevkit/VOC2012/', aug_path='../../../VOCdevkit/VOC2012/SegmentationClass/')
voc2012.read_train_images()
voc2012.read_train_labels()
voc2012.read_val_images()
voc2012.read_val_labels()

def binarylab(labels):
  x = np.zeros([224, 224, 21])
  for i in range(224):
    for j in range(224):
      if labels[i][j]>0:
          x[i, j, labels[i][j]] = 1
  return x

trainlabels =np.array([binarylab(my_labels) for my_labels in tqdm(voc2012.train_labels)])
vallabels = np.array([binarylab(my_labels) for my_labels in tqdm(voc2012.val_labels)])
save_h5("../voc2012_train.h5", voc2012.train_images, trainlabels)
save_h5("../voc2012_val.h5", voc2012.val_images, vallabels)