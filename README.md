## TICNN & PRCNN

### Overview
This repository contains the implementation code of **TICNN**: [*TICNN*](http://staff.ustc.edu.cn/~xinmei/publications_pdf/2016/Transform-Invariant%20Convolutional%20Neural%20Networks%20for%20Image%20Classification%20and%20Search.pdf) .

Convolutional neural networks (CNNs) have achieved state-of-the-art results on many visual recognition tasks. However, current CNN models still exhibit a poor ability to be invariant to spatial transformations of images. Intuitively, with sufficient layers and parameters, hierarchical combinations of convolution (matrix multiplication and non-linear activation) and pooling operations should be able to learn a robust mapping from transformed input images to transform-invariant representations. In this paper, we propose randomly transforming (rotation, scale, and translation) feature maps of CNNs during the training stage. This prevents complex dependencies of specific rotation, scale, and translation levels of training images in CNN models. Rather, each convolutional kernel learns to detect a feature that is generally helpful for producing the transform-invariant answer given the combinatorially large variety of transform levels of its input feature maps. In this way, we do not require any extra training supervision or modification to the optimization process and training images. We show that random transformation provides significant improvements of CNNs on many benchmark tasks, including small-scale image recognition, large-scale image recognition, and image retrieval.

and **PRCNN**: [*PRCNN*](https://arxiv.org/abs/1911.12682).

Convolutional Neural Networks (CNNs) have demonstrated state-of-the-art performance on many visual recognition tasks. However, the combination of convolution and pooling operations only shows invariance to small local location changes in meaningful objects in input. Sometimes, such networks are trained using data augmentation to encode this invariance into the parameters, which restricts the capacity of the model to learn the content of these objects. A more efficient use of the parameter budget is to encode rotation or translation invariance into the model architecture, which relieves the model from the need to learn them. To enable the model to focus on learning the content of objects other than their locations, we propose to conduct patch ranking of the feature maps before feeding them into the next layer. When patch ranking is combined with convolution and pooling operations, we obtain consistent representations despite the location of meaningful objects in input. We show that the patch ranking module improves the performance of the CNN on many benchmark tasks, including MNIST digit recognition, large-scale image recognition, and image retrieval.

### TICNN Source code
include/caffe/layers/random_transform_layer.hpp

src/caffe/layers/random_trans_layer.cpp

src/caffe/layers/random_trans_layer.cu

### PRCNN Source code
include/caffe/layers/patch_rank_layer.hpp

src/caffe/layers/patch_rank_layer.cpp

src/caffe/layers/patch_rank_layer.cu


## License and Citation

Please cite this paper if you are interested in the random_trans_layer:

    @inproceedings{shen-mm16,
     author = {Shen, Xu and Tian, Xinmei and He, Anfeng and Sun, Shaoyan and Tao, Dacheng},
     title = {Transform-Invariant Convolutional Neural Networks for Image Classification and Search},
     booktitle = {ACM MM},
     year = {2016},
     pages = {1345--1354}
    } 
    
Please cite this paper if you are interested in the patch_rank_layer:

    @inproceedings{shen-aaai17,
     author = {Shen, Xu and Tian, Xinmei and Sun, Shaoyan and Tao, Dacheng},
     title = {Patch Reordering: A Novel Way to Achieve Rotation and Translation Invariance in Convolutional Neural Networks},
     booktitle = {AAAI},
     year = {2017}
    } 


Caffe is released under the [BSD 2-Clause license](https://github.com/BVLC/caffe/blob/master/LICENSE).
The BVLC reference models are released for unrestricted use.

Please cite Caffe in your publications if it helps your research:

    @article{jia2014caffe,
      Author = {Jia, Yangqing and Shelhamer, Evan and Donahue, Jeff and Karayev, Sergey and Long, Jonathan and Girshick, Ross and Guadarrama, Sergio and Darrell, Trevor},
      Journal = {arXiv preprint arXiv:1408.5093},
      Title = {Caffe: Convolutional Architecture for Fast Feature Embedding},
      Year = {2014}
    }
