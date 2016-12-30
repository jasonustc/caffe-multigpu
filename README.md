Here is the linux/windows compatible version of caffe forked from https://github.com/BVLC/caffe in 04/10/2016 . Multi-GPU is
supported in this version.

I also have a talk on brief introduction of Deep Learning, [part1](http://v.youku.com/v_show/id_XMTYyMTk1NDU2MA==.html),[part2](http://v.youku.com/v_show/id_XMTYyMTk2MTEwOA==.html), [slides](http://pan.baidu.com/s/1hrMmyS8).

## Windows
Tools:

1. Visual Studio 2013

2. Cuda 7.5 (**you should install cuda after the installation of Visual Studio 2013 to incorporate cuda vs integration into VS**)

3. OpenCV 2.4.9

4. Boost

Steps:

1. Copy folder \$3rdparty (http://pan.baidu.com/s/1ge3nKRp) and \$bin (http://pan.baidu.com/s/1jIyEjKq) to the caffe root directory

2. Configure the environment variables: \$BOOST_1_56_0, \$OPENCV_2_4_9

3. Compile the caffe.sln in VS2013

Notes:

1. Currently Caffe works with cuDNN_v3 or cuDNN_v4 (**The current settings in caffe.sln do not use cuDNN**)

You need copy More details at https://github.com/BVLC/caffe/tree/windows

## Linux

Please follow the official tutorials here: http://caffe.berkeleyvision.org/installation.html 

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
     title = {Patch	Reordering:	a Novel	Way	to Achieve Rotation	and	Translation	Invariance in Convolutional Neural Networks},
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
