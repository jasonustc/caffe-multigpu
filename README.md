Here is the windows version of caffe forked in 04/10/2016. Multi-GPU is
supported in this version.

Tools:
1. Visual Studio 2013
2. Cuda 7.5
3. OpenCV 2.4.9

Steps:
1. Copy folder \$3rdparty and \$bin to the caffe root directory
2. Configure the environment variables: \$BOOST_1_56_0, \$OPENCV_2_4_9
3. Compile the caffe.sln in VS2013

Notes:
1. Currently Caffe works with cuDNN_v3 or cuDNN_v4
2. You need to compile cudnn_*_.cu files firstly manually, then compile the project (I don't know why too...)

You need copy 
More details at https://github.com/BVLC/caffe/tree/windows