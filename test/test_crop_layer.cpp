#include <cstring>
#include <vector>

#include "cuda_runtime.h"
#include "gtest/gtest.h"
#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/test/test_gradient_check_util.hpp"
#include "caffe/test/test_caffe_main.hpp"
#include "caffe/layers/crop_layer.hpp"

namespace caffe{
	template <typename Dtype>
	class CropLayerTest{
	public:
		CropLayerTest(){
			this->SetUp();
		}

		void TestSetUp(){
			shared_ptr<Layer<Dtype>> layer(new CropLayer<Dtype>(layer_param_));
			layer->SetUp(bottom_, top_);
			CHECK_EQ(top_[0]->shape(0), 5);
			CHECK_EQ(top_[0]->shape(1), 2);
			CHECK_EQ(top_[0]->shape(2), 2);
			CHECK_EQ(top_[0]->shape(3), 2);
		}

		void TestForward(Caffe::Brew mode){
			shared_ptr<Layer<Dtype>> layer(new CropLayer<Dtype>(layer_param_));
			Caffe::set_mode(mode);
			layer->SetUp(bottom_, top_);
			layer->Forward(bottom_, top_);
			if (mode == Caffe::CPU){
				bottom_[0]->ToTxt("bottom_cpu");
				top_[0]->ToTxt("top_cpu");
			}
			else{
				bottom_[0]->ToTxt("bottom_gpu");
				top_[0]->ToTxt("top_gpu");
			}
		}


		void TestGradients(Caffe::Brew mode){
			Caffe::set_mode(mode);
			GradientChecker<Dtype> checker(0.001, 0.0001);
			CropLayer<Dtype> layer(layer_param_);
			CHECK_GT(top_.size(), 0) << "Exhaustive mode requires at least one top blob.";
			checker.CheckGradientExhaustive(&layer, bottom_, top_, 0);
		}


	protected:
		void SetUp(){
			input_ = new Blob<Dtype>();
			output_ = new Blob<Dtype>();
			vector<int> x_shape{ 5, 2, 4, 3};
			input_->Reshape(x_shape);
			Dtype* x_data = input_->mutable_cpu_data();
			for (int c = 0; c < 10; ++c){
				for (int j = 0; j < 4; j++){
					for (int k = 0; k < 3; ++k){
						x_data[c * 12 + j * 3 + k] = c * 0.1 + j * 0.3 + k * 0.2;
					}
				}
			}

			//set bottom && top
			bottom_.push_back(input_);
			top_.push_back(output_);
			layer_param_.mutable_crop_param()->set_axis(2);
			layer_param_.mutable_crop_param()->add_crop_size(2);
			layer_param_.mutable_crop_param()->add_crop_size(2);
			layer_param_.mutable_crop_param()->set_random(true);
			layer_param_.mutable_crop_param()->add_offset(1);
			layer_param_.mutable_crop_param()->add_offset(1);
		}

		Blob<Dtype>* input_;
		Blob<Dtype>* output_;

		vector<Blob<Dtype>*> bottom_;
		vector<Blob<Dtype>*> top_;

		LayerParameter layer_param_;
	};
}

int main(int argc, char** argv){
	::google::InitGoogleLogging(*argv);
	FLAGS_logtostderr = true;
	caffe::CropLayerTest<float> test;
//	test.TestSetUp();
//	test.TestForward(caffe::Caffe::CPU);
	test.TestForward(caffe::Caffe::GPU);
	test.TestGradients(caffe::Caffe::CPU);
	test.TestGradients(caffe::Caffe::GPU);
	return 0;
}