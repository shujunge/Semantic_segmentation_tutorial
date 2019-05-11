## The semantic image segmentation on the PASCAL VOC dataset
The original post can be found [here](https://github.com/DrSleep/tensorflow-deeplab-resnet), by @DrSleep.<br />
The code had been tested with python2.7.

### 1. preparation
The pretrained "DeepLab-ResNet-TensorFlow" model should be download to the "./trained_model/" directory with the following link [deeplab_resnet](https://drive.google.com/open?id=0B_rootXHuswsVGY4ZFRydXJBR1E)

### 2. test
to test a single image, we can run the following
```bash
python inference.py ./test_image/test1.jpg ./trained_model/deeplab_resnet.ckpt 
```
The result image file will use saved in the directory "./output/". The color scheme of the classification/segmentation can be found at "./test_image/colour_scheme.png".
