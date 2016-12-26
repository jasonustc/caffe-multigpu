#include <cstring>
#include <vector>

#include "cuda_runtime.h"
#include "gtest/gtest.h"
#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/test/test_gradient_check_util.hpp"
#include "caffe/test/test_caffe_main.hpp"
#include "caffe/layers/euclidean_loss_layer.hpp"

namespace caffe{
	template <typename Dtype>
	class EuclideanLossLayerTest{
	public:
		EuclideanLossLayerTest(){
			this->SetUp();
		}

		void TestSetUp(){
			shared_ptr<Layer<Dtype>> layer(new EuclideanLossLayer<Dtype>(layer_param_));
			layer->SetUp(bottom_, top_);
		}

		void TestForward(Caffe::Brew mode){
			shared_ptr<Layer<Dtype>> layer(new EuclideanLossLayer<Dtype>(layer_param_));
			Caffe::set_mode(mode);
			layer->SetUp(bottom_, top_);
			bottom_[0]->ToTxt("bottom0");
			bottom_[1]->ToTxt("bottom1");
			bottom_[2]->ToTxt("bottom2");
			layer->Forward(bottom_, top_);
			if (mode == Caffe::CPU){
				top_[0]->ToTxt("top_cpu");
			}
			else{
				top_[0]->ToTxt("top_gpu");
			}
		}


		void TestGradients(Caffe::Brew mode){
			Caffe::set_mode(mode);
			GradientChecker<Dtype> checker(0.001, 0.0001);
			EuclideanLossLayer<Dtype> layer(layer_param_);
			CHECK_GT(top_.size(), 0) << "Exhaustive mode requires at least one top blob.";
			checker.CheckGradientExhaustive(&layer, bottom_, top_, 0);
			checker.CheckGradientExhaustive(&layer, bottom_, top_, 1);
		}


	protected:
		void SetUp(){
			input0_ = new Blob<Dtype>();
			input1_ = new Blob<Dtype>();
			input2_ = new Blob<Dtype>();
			output_ = new Blob<Dtype>();
			vector<int> x_shape{ 5, 2, 2, 2 };
			input0_->Reshape(x_shape);
			input1_->Reshape(x_shape);
			vector<int> y_shape{ 5, 2 };
			input2_->Reshape(y_shape);
			Dtype* x_data = input0_->mutable_cpu_data();
			Dtype* x1_data = input1_->mutable_cpu_data();
			for (int i = 0; i < 5; ++i){
				for (int c = 0; c < 2; ++c){
					for (int j = 0; j < 6; j++){
						x_data[i * 8 + c * 4 + j] = c * 0.1 + j * 0.3 + i;
						x1_data[i * 8 + c * 4 + j] = c * 0.3 + j * 0.1 + i;
					}
				}
			}
			Dtype* ind_data = input2_->mutable_cpu_data();
			for (int i = 0; i < 5; ++i){
				for (int j = 0; j < 2; ++j){
					ind_data[i * 2 + j] = (i * 2 + j) % 2 == 0;
				}
			}
			input2_->ToTxt("ind");

			//set bottom && top
			bottom_.push_back(input0_);
			bottom_.push_back(input1_);
			bottom_.push_back(input2_);
			top_.push_back(output_);
			layer_param_.mutable_scale_param()->set_axis(0);
			layer_param_.mutable_scale_param()->set_num_axes(2);
		}

		Blob<Dtype>* input0_;
		Blob<Dtype>* input1_;
		Blob<Dtype>* input2_;
		Blob<Dtype>* output_;

		vector<Blob<Dtype>*> bottom_;
		vector<Blob<Dtype>*> top_;

		LayerParameter layer_param_;
	};
}

int main(int argc, char** argv){
	::google::InitGoogleLogging(*argv);
	FLAGS_logtostderr = true;
	caffe::EuclideanLossLayerTest<float> test;
	test.TestSetUp();
	test.TestForward(caffe::Caffe::CPU);
	test.TestForward(caffe::Caffe::GPU);
	test.TestGradients(caffe::Caffe::CPU);
	test.TestGradients(caffe::Caffe::GPU);
	return 0;
}