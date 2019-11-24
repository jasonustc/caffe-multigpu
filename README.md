# Continous Dropout

### Overview
This repository contains the implementation code of Continuous Dropout: [*Continuous Dropout*](http://staff.ustc.edu.cn/~xinmei/publications_pdf/2017/Continuous%20Dropout.pdf).

We extend the traditional binary dropout
to continuous dropout. On the one hand, continuous dropout is
considerably closer to the activation characteristics of neurons
in the human brain than traditional binary dropout. On the
other hand, we demonstrate that continuous dropout has the
property of avoiding the co-adaptation of feature detectors, which
suggests that we can extract more independent feature detectors
for model averaging in the test stage.

### Source code
include/caffe/layers/dropout_layer.hpp

src/caffe/layers/dropout_layer.cpp

src/caffe/layers/dropout_layer.cu


### Citation
Please cite this paper in your publications if it helps your research:

```
@inproceedings{xu2018cd,
  author={X. {Shen} and X. {Tian} and T. {Liu} and F. {Xu} and D. {Tao}},
  journal={IEEE Transactions on Neural Networks and Learning Systems},
  title={Continuous Dropout},
  year={2018},
  pages={3926-3937}
}
```

Caffe is released under the [BSD 2-Clause license](https://github.com/BVLC/caffe/blob/master/LICENSE).
The BVLC reference models are released for unrestricted use.

Please cite Caffe in your publications if it helps your research:
```
    @article{jia2014caffe,
      Author = {Jia, Yangqing and Shelhamer, Evan and Donahue, Jeff and Karayev, Sergey and Long, Jonathan and Girshick, Ross and Guadarrama, Sergio and Darrell, Trevor},
      Journal = {arXiv preprint arXiv:1408.5093},
      Title = {Caffe: Convolutional Architecture for Fast Feature Embedding},
      Year = {2014}
    }
```
