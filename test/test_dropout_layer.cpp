#include <cstring>
#include <vector>

#include "cuda_runtime.h"
#include "gtest/gtest.h"
#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/test/test_gradient_check_util.hpp"
#include "caffe/test/test_caffe_main.hpp"
#include "caffe/layers/dropout_layer.hpp"

namespace caffe{
	template <typename Dtype>
	class DropoutLayerTest{
	public:
		DropoutLayerTest() : input_(new Blob<Dtype>()), output_(new Blob<Dtype>()){
			this->SetUp();
		}

		void TestSetUp(){
			shared_ptr<Layer<Dtype>> layer(new DropoutLayer<Dtype>(layer_param_));
			layer->SetUp(bottom_, top_);
			CHECK_EQ(top_[0]->shape(0), 5);
			CHECK_EQ(top_[0]->shape(1), 2);
			CHECK_EQ(top_[0]->shape(2), 2);
		}

		void TestForward(Caffe::Brew mode){
			shared_ptr<Layer<Dtype>> layer(new DropoutLayer<Dtype>(layer_param_));
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
			GradientChecker<Dtype> checker(0.01, 0.001);
			DropoutLayer<Dtype> layer(layer_param_);
			CHECK_GT(top_.size(), 0) << "Exhaustive mode requires at least one top blob.";
			checker.CheckGradientExhaustive(&layer, bottom_, top_, 0);
		}


	protected:
		void SetUp(){
			vector<int> x_shape{ 5, 2, 2 };
			input_->Reshape(x_shape);
			Dtype* x_data = input_->mutable_cpu_data();
			for (int c = 0; c < 5; ++c){
				for (int j = 0; j < 4; j++){
					x_data[c * 4 + j] = c * 0.1 + j * 0.3;
				}
			}

			//set bottom && top
			bottom_.push_back(input_);
			top_.push_back(output_);
			propagate_down_.resize(1, true);
			layer_param_.mutable_dropout_param()->set_num_axes(0);
			layer_param_.mutable_dropout_param()->set_drop_type(DropoutParameter_DropType_BERNOULLI);
			layer_param_.mutable_dropout_param()->set_sigma(0.1);
		}

		Blob<Dtype>* input_;
		Blob<Dtype>* output_;

		vector<Blob<Dtype>*> bottom_;
		vector<Blob<Dtype>*> top_;

		vector<bool> propagate_down_;

		LayerParameter layer_param_;
	};
}

int main(int argc, char** argv){
	::google::InitGoogleLogging(*argv);
	FLAGS_logtostderr = true;
	caffe::DropoutLayerTest<float> test;
	test.TestSetUp();
	test.TestForward(caffe::Caffe::CPU);
	test.TestForward(caffe::Caffe::GPU);
	test.TestGradients(caffe::Caffe::CPU);
	test.TestGradients(caffe::Caffe::GPU);
	return 0;
}