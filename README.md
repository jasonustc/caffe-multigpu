Here is the linux/windows compatible version of caffe forked from https://github.com/BVLC/caffe in 04/10/2016 . Multi-GPU is
supported in this version.

## Windows
Tools:

1. Visual Studio 2013

2. Cuda 7.5

3. OpenCV 2.4.9

4. Boost

Steps:

1. Copy folder \$3rdparty (http://pan.baidu.com/s/1ge3nKRp) and \$bin (http://pan.baidu.com/s/1jIyEjKq) to the caffe root directory

2. Configure the environment variables: \$BOOST_1_56_0, \$OPENCV_2_4_9

3. Compile the caffe.sln in VS2013

Notes:

1. Currently Caffe works with cuDNN_v3 or cuDNN_v4 (**The current settings in caffe.sln do not use cuDNN**)

2. You need to compile cudnn_*_.cu files firstly manually, then compile the project (I don't know why too...)

You need copy 
More details at https://github.com/BVLC/caffe/tree/windows

## Linux

Please follow the official tutorials here: http://caffe.berkeleyvision.org/installation.html 

## License and Citation

Caffe is released under the [BSD 2-Clause license](https://github.com/BVLC/caffe/blob/master/LICENSE).
The BVLC reference models are released for unrestricted use.

Please cite Caffe in your publications if it helps your research:

    @article{jia2014caffe,
      Author = {Jia, Yangqing and Shelhamer, Evan and Donahue, Jeff and Karayev, Sergey and Long, Jonathan and Girshick, Ross and Guadarrama, Sergio and Darrell, Trevor},
      Journal = {arXiv preprint arXiv:1408.5093},
      Title = {Caffe: Convolutional Architecture for Fast Feature Embedding},
      Year = {2014}
    }
