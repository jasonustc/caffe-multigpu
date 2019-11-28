## TICNN & PRCNN

### Overview
This repository contains the implementation code of TICNN: [*TICNN*](http://staff.ustc.edu.cn/~xinmei/publications_pdf/2016/Transform-Invariant%20Convolutional%20Neural%20Networks%20for%20Image%20Classification%20and%20Search.pdf) 

and PRCNN: [*PRCNN*](https://aaai.org/ocs/index.php/AAAI/AAAI17/paper/view/14674/14441).


### TICNN Source code
include/caffe/layers/random_trans_layer.hpp

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
     title = {Patch Reordering: a Novel Way to Achieve Rotation and Translation Invariance in Convolutional Neural Networks},
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
