// Copyright 2014 BVLC and contributors.
// Author: Xu Shen
// This cpp file can be used to extract feature of one single image loaded from
// local disk.

#include <stdio.h>  // for snprintf
#include <string>
#include <vector>
#include "caffe/common.hpp"

#include "deep_aesth.hpp"

DEFINE_string(net_file, "imagenet_val.prototxt",
	"net model file for feature extraction");
DEFINE_string(model_file, "bvlc_reference_caffenet.caffemodel",
	"trained parameter model");
DEFINE_bool(sqrt, false,
	"Optional; if we need to do root square on extracted features.");
DEFINE_bool(l2_norm, false,
	"Optional; if we need to do L2 normalization on extracted features.");
DEFINE_string(svm_model, "model_227.xml",
	"The svm model to classify aesthetic photos.");


int main(int argc, char** argv){
	float score;
	DeepAesth<float> deep_aesth(FLAGS_net_file, FLAGS_model_file);
	deep_aesth.LoadImage("newHigh\\27.jpg");
	deep_aesth.GetFeat("fc7");
	score = deep_aesth.GetScore();
	std::cout << score << "\n";
	deep_aesth.LoadImage("newHigh\\34.jpg");
	deep_aesth.GetFeat("fc7");
	score = deep_aesth.GetScore();
	std::cout << score << "\n";
	deep_aesth.LoadImage("newHigh\\41.jpg");
	deep_aesth.GetFeat("fc7");
	score = deep_aesth.GetScore();
	std::cout << score << "\n";
	deep_aesth.LoadImage("newHigh\\51.jpg");
	deep_aesth.GetFeat("fc7");
	score = deep_aesth.GetScore();
	std::cout << score << "\n";
	deep_aesth.LoadImage("newHigh\\64.jpg");
	deep_aesth.GetFeat("fc7");
	score = deep_aesth.GetScore();
	std::cout << score << "\n";
	deep_aesth.LoadImage("newHigh\\66.jpg");
	deep_aesth.GetFeat("fc7");
	score = deep_aesth.GetScore();
	std::cout << score << "\n";
	deep_aesth.LoadImage("newHigh\\67.jpg");
	deep_aesth.GetFeat("fc7");
	score = deep_aesth.GetScore();
	std::cout << score << "\n";
	deep_aesth.LoadImage("temp.jpg");
	deep_aesth.GetFeat("fc7");
	score = deep_aesth.GetScore();
	std::cout << score << "\n";
	deep_aesth.LoadImage("test.png");
	deep_aesth.GetFeat("fc7");
	score = deep_aesth.GetScore();
	std::cout << score << "\n";
	deep_aesth.LoadImage("view\\134.jpg");
	deep_aesth.GetFeat("fc7");
	score = deep_aesth.GetScore();
	std::cout << score << "\n";
	return 0;
}
