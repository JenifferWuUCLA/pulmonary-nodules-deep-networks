# Deep Learning Tutorial for Pulmonary Nodules Deep Networks, using Caffe

## 天池医疗AI大赛[第一季]：Caffe训练基于卷积神经网络的肺结节分类器
> ##### @author Jeniffer Wu

#### 服务器数据集分布：

|            | Total num   | TRAIN num  | TEST num   |
|------------|------------ |------------|------------|
| Caffe      | 4908        | 3504       | 1404       | 

Current config for 8 GPUs with 32 mini-batch size each.

The layer names are designed to match MSRA released pre-trained models to allow for finetuning. You may need to enable/disable bias on `conv1` to use these prototxts with some pretrained models. 

Running the ResNet-50 as-is gets a few percent lower accuracy than MSRA if done without random reshape & crop. 

Pulmonary-nodules-deep-networks relies on external machine learning libraries through a very generic and flexible API. At the moment it has support for:

- the deep learning library [Caffe](https://github.com/BVLC/caffe)
- the deep learning and other usages library [Tensorflow](https://tensorflow.org)

#### Machine Learning functionalities per library (current):

|            | Training | Prediction | Classification | Object Detection | Segmentation | Regression | Autoencoder |
|------------|----------|------------|----------------|-----------|-----------|------------|-------------|
| Caffe      | Y        | Y          | Y              | Y         |   Y       |   Y        | Y           |
| Tensorflow | N        | Y          | Y              | N         |   N       |   N        | N           |

#### GPU support per library

|            | Training | Prediction |
|------------|----------|------------|
| Caffe      | Y        | Y          |
| Tensorflow | Y        | Y          |

#### Input data support per library (current):

|            | CSV | SVM | Text words | Text characters | Images |
|------------|-----|-----|------------|-----------------|--------|
| Caffe      | Y   | Y   | Y          | Y               | Y      |
| Tensorflow | N   | N   | N          | N               | Y      |

#### Main functionalities

Pulmonary-nodules-deep-networks implements support for supervised and unsupervised deep learning of images, text and other data, with focus on simplicity and ease of use, test and connection into existing applications. It supports classification, object detection, segmentation, regression, autoencoders, ...

##### Caffe Dependencies

- CUDA 8 or 7.5 is recommended for GPU mode.
- BLAS via ATLAS, MKL, or OpenBLAS.
- [protobuf](https://github.com/google/protobuf)
- IO libraries hdf5, leveldb, snappy, lmdb

#### Tensorflow Dependencies

- Cmake > 3
- [Bazel](https://www.bazel.io/versions/master/docs/install.html#install-on-ubuntu)

##### Caffe version

By default Pulmonary-nodules-deep-networks automatically relies on a modified version of Caffe, https://github.com/beniz/caffe/tree/master
This version includes many improvements over the original Caffe, such as sparse input data support, exception handling, class weights, object detection, segmentation, and various additional losses and layers.

##### Implementation

The code makes use of C++ policy design for modularity, performance and putting the maximum burden on the checks at compile time. The implementation uses many features from C++11.

##### Models

|                          | Caffe | Tensorflow | Source        | Top-1 Accuracy (ImageNet) |
|--------------------------|-------|------------|---------------|---------------------------|
| AlexNet                  | Y     | N          | BVLC          |          57.1%                 |
| SqueezeNet               | Y     | N          | DeepScale              |       59.5%                    | 
| Inception v1 / GoogleNet | [Y](https://deepdetect.com/models/ggnet/bvlc_googlenet.caffemodel)     | [Y](https://deepdetect.com/models/tf/inception_v1.pb)          | BVLC / Google |             67.9%              |
| Inception v2             | N     | [Y](https://deepdetect.com/models/tf/inception_v2.pb)          | Google        |     72.2%                      |
| Inception v3             | N     | [Y](https://deepdetect.com/models/tf/inception_v3.pb)          | Google        |         76.9%                  |
| Inception v4             | N     | [Y](https://deepdetect.com/models/tf/inception_v4.pb)          | Google        |         80.2%                  |
| ResNet 50                | [Y](https://deepdetect.com/models/resnet/ResNet-50-model.caffemodel)     | [Y](https://deepdetect.com/models/tf/resnet_v1_50/resnet_v1_50.pb)          | MSR           |      75.3%                     |
| ResNet 101               | [Y](https://deepdetect.com/models/resnet/ResNet-101-model.caffemodel)     | [Y](https://deepdetect.com/models/tf/resnet_v1_101/resnet_v1_101.pb)          | MSR           |        76.4%                   |
| ResNet 152               | [Y](https://deepdetect.com/models/resnet/ResNet-152-model.caffemodel)     | [Y](https://deepdetect.com/models/tf/resnet_v1_152/resnet_v1_152.pb)         | MSR           |               77%            |
| Inception-ResNet-v2      | N     | [Y](https://deepdetect.com/models/tf/inception_resnet_v2.pb)          | Google        |       79.79%                    |
| VGG-16                   | [Y](https://deepdetect.com/models/vgg_16/VGG_ILSVRC_16_layers.caffemodel)     | [Y](https://deepdetect.com/models/tf/vgg_16/vgg_16.pb)          | Oxford        |               70.5%            |
| VGG-19                   | [Y](https://deepdetect.com/models/vgg_19/VGG_ILSVRC_19_layers.caffemodel)     | [Y](https://deepdetect.com/models/tf/vgg_19/vgg_19.pb)          | Oxford        |               71.3%            |
| ResNext 50                | [Y](https://deepdetect.com/models/resnext/resnext_50)     | N          | https://github.com/terrychenism/ResNeXt           |      76.9%                     |
| ResNext 101                | [Y](https://deepdetect.com/models/resnext/resnext_101)     | N          | https://github.com/terrychenism/ResNeXt           |      77.9%                     |
| ResNext 152               | [Y](https://deepdetect.com/models/resnext/resnext_152)     | N          | https://github.com/terrychenism/ResNeXt           |      78.7%                     |
| DenseNet-121                   | [Y](https://deepdetect.com/models/densenet/densenet_121_32/)     | N          | https://github.com/shicai/DenseNet-Caffe        |               74.9%            |
| DenseNet-161                   | [Y](https://deepdetect.com/models/densenet/densenet_161_48/)     | N          | https://github.com/shicai/DenseNet-Caffe        |               77.6%            |
| DenseNet-169                   | [Y](https://deepdetect.com/models/densenet/densenet_169_32/)     | N          | https://github.com/shicai/DenseNet-Caffe        |               76.1%            |
| DenseNet-201                   | [Y](https://deepdetect.com/models/densenet/densenet_201_32/)     | N          | https://github.com/shicai/DenseNet-Caffe        |               77.3%            |
| VOC0712 (object detection) | [Y](https://deepdetect.com/models/voc0712_dd.tar.gz) | N | https://github.com/weiliu89/caffe/tree/ssd | 71.2 mAP |

More models:

- List of free, even for commercial use, deep neural nets for image classification, and character-based convolutional nets for text classification: http://www.deepdetect.com/applications/list_models/

#### Templates

Pulmonary-nodules-deep-networks comes with a built-in system of neural network templates (Caffe backend only at the moment). This allows the creation of custom networks based on recognized architectures, for images, text and data, and with much simplicity.

Usage:
- specify `template` to use, from `mlp`, `convnet` and `resnet`
- specify the architecture with the `layers` parameter:
  - for `mlp`, e.g. `[300,100,10]`
  - for `convnet`, e.g. `["1CR64","1CR128","2CR256","1024","512"], where the main pattern is `xCRy` where `y` is the number of outputs (feature maps), `CR` stands for Convolution + Activation (with `relu` as default), and `x` specifies the number of chained `CR` blocks without pooling. Pooling is applied between all `xCRy`
- for `resnets`:
   - with images, e.g. `["Res50"]` where the main pattern is `ResX` with X the depth of the Resnet
   - with character-based models (text), use the `xCRy` pattern of convnets instead, with the main difference that `x` now specifies the number of chained `CR` blocks within a resnet block
   - for Resnets applied to CSV or SVM (sparse data), use the `mlp` pattern. In this latter case, at the moment, the `resnet` is built with blocks made of two layers for each specified layer after the first one. Here is an example: `[300,100,10]` means that a first hidden layer of size `300` is applied followed by a `resnet` block made of two `100` fully connected layer, and another block of two `10` fully connected layers. This is subjected to future changes and more control.

### Authors
Pulmonary-nodules-deep-networks is designed and implemented by Yingyi Wu  <yywu@szucla.org>.

#### Default build with Caffe
For compiling along with Caffe:
```
mkdir build
cd build
cmake ..
make
```

If you are building for one or more GPUs, you may need to add CUDA to your ld path:
```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64
```

If you would like to build with cuDNN, your `cmake` line should be:
```
cmake .. -DUSE_CUDNN=ON
```

To target the build of underlying Caffe to a specific CUDA architecture (e.g. Pascal), you can use:
```
cmake .. -DCUDA_ARCH="-gencode arch=compute_61,code=sm_61"
```

If you would like a CPU only build, use:
```
cmake .. -DUSE_CPU_ONLY=ON
```

If you would like to constrain Caffe to CPU only, use:
```
cmake .. -DUSE_CAFFE_CPU_ONLY=ON
```

#### Build with Tensorflow support
First you must install [Bazel](https://www.bazel.io/versions/master/docs/install.html#install-on-ubuntu) and Cmake with version > 3.

And other dependencies:
```
sudo apt-get install python-numpy swig python-dev python-wheel unzip
```

If you would like to build with Tensorflow, include the `-DUSE_TF=ON` paramter to `cmake`:
```
cmake .. -DUSE_TF=ON
```

If you would like to constrain Tensorflow to CPU, use:
```
cmake .. -DUSE_TF=ON -DUSE_TF_CPU_ONLY=ON
```

You can combine with XGBoost support with:
```
cmake .. -DUSE_TF=ON -DUSE_XGBOOST=ON
```

### References

- Caffe (https://github.com/BVLC/caffe)
