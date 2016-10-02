#include <cstring>
#include <vector>

#include "cuda_runtime.h"
#include "gtest/gtest.h"
#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/test/test_gradient_check_util.hpp"
#include "caffe/test/test_caffe_main.hpp"
#include "caffe/layers/switch_layer.hpp"

namespace caffe{
	template <typename Dtype>
	class SwitchLayerTest{
	public:
		SwitchLayerTest() : 
			input1_(new Blob<Dtype>()), 
			input2_(new Blob<Dtype>()), 
			input3_(new Blob<Dtype>()), 
			bottom_index_(new Blob<Dtype>()),
			output_(new Blob<Dtype>()),
			switch_index_(new Blob<Dtype>()){
			this->SetUp();
		}

		void TestSetUp(){
			shared_ptr<Layer<Dtype>> layer(new SwitchLayer<Dtype>(layer_param_));
			layer->SetUp(bottom_, top_);
			CHECK_EQ(top_[0]->shape(0), 5);
			CHECK_EQ(top_[0]->shape(1), 2);
			CHECK_EQ(top_[0]->shape(2), 2);
		}

		void TestForward(Caffe::Brew mode){
			shared_ptr<Layer<Dtype>> layer(new SwitchLayer<Dtype>(layer_param_));
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
			SwitchLayer<Dtype> layer(layer_param_);
			CHECK_GT(top_.size(), 0) << "Exhaustive mode requires at least one top blob.";
			checker.CheckGradientExhaustive(&layer, bottom_, top_);
		}


	protected:
		void SetUp(){
			vector<int> x_shape{ 5, 2, 2 };
			input1_->Reshape(x_shape);
			input2_->Reshape(x_shape);
			input3_->Reshape(x_shape);
			Dtype* x1_data = input1_->mutable_cpu_data();
			Dtype* x2_data = input2_->mutable_cpu_data();
			Dtype* x3_data = input3_->mutable_cpu_data();
			for (int c = 0; c < 5; ++c){
				for (int j = 0; j < 4; j++){
					x1_data[c * 4 + j] = c * 0.1 + j * 0.3;
					x2_data[c * 4 + j] = c * 0.2 + j * 0.2;
					x3_data[c * 4 + j] = c * 0.3 + j * 0.2;
				}
			}

			//set bottom && top
			bottom_index_->Reshape(vector<int>(1, 1));
			bottom_index_->mutable_cpu_data()[0] = 1;
			bottom_.push_back(input1_);
			bottom_.push_back(input2_);
			bottom_.push_back(input3_);
			bottom_.push_back(bottom_index_);
			top_.push_back(output_);
			top_.push_back(switch_index_);
			layer_param_.mutable_switch_param()->set_has_bottom_index(true);
		}

		Blob<Dtype>* input1_;
		Blob<Dtype>* input2_;
		Blob<Dtype>* input3_;
		Blob<Dtype>* bottom_index_;
		Blob<Dtype>* output_;
		Blob<Dtype>* switch_index_;

		vector<Blob<Dtype>*> bottom_;
		vector<Blob<Dtype>*> top_;


		LayerParameter layer_param_;
	};
}

int main(int argc, char** argv){
	::google::InitGoogleLogging(*argv);
	FLAGS_logtostderr = true;
	caffe::SwitchLayerTest<float> test;
	test.TestSetUp();
	test.TestForward(caffe::Caffe::CPU);
	test.TestForward(caffe::Caffe::GPU);
	test.TestGradients(caffe::Caffe::CPU);
	test.TestGradients(caffe::Caffe::GPU);
	return 0;
}