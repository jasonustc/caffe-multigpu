### Basic Info
- input: image + 5 local blocks  

- feature: bvlc_reference_caffenet.caffemodel, fc7

- related_materials: http://pan.baidu.com/s/1o7Pz3vG, please copy "model_227.xml", "image_val.prototxt", "bvlc_reference_caffenet.caffemodel" into the same folder with the executable file.

- evaluate_dataset: AVA ( 19308 train, 19308 test)

- performance:  Accuracy: 83.5%

- license: unrestricted

### Related code are here:
- [liblinear](https://github.com/jasonustc/caffe-multigpu/tree/deep_aesth/src/liblinear)
- [feature_extractor](https://github.com/jasonustc/caffe-multigpu/blob/deep_aesth/include/deep_aesth.hpp)
- [main](https://github.com/jasonustc/caffe-multigpu/blob/deep_aesth/tools/deep_aesth.cpp)
