#include <cstring>
#include <vector>

#include "cuda_runtime.h"
#include "gtest/gtest.h"
#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/test/test_gradient_check_util.hpp"
#include "caffe/test/test_caffe_main.hpp"
#include "caffe/layers/semi_hinge_loss_layer.hpp"

namespace caffe{
	template <typename Dtype>
	class SemiHingeLossLayerTest{
	public:
		SemiHingeLossLayerTest() : 
			input_0(new Blob<Dtype>()), 
			input_1(new Blob<Dtype>()),
			input_2(new Blob<Dtype>()),
			input_3(new Blob<Dtype>()),
			output_(new Blob<Dtype>())
		{
			this->SetUp();
		}

		void TestSetUp(){
			shared_ptr<Layer<Dtype>> layer(new SemiHingeLossLayer<Dtype>(layer_param_));
			layer->SetUp(bottom_, top_);
		}

		void TestForward(Caffe::Brew mode){
			shared_ptr<Layer<Dtype>> layer(new SemiHingeLossLayer<Dtype>(layer_param_));
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
			SemiHingeLossLayer<Dtype> layer(layer_param_);
			CHECK_GT(top_.size(), 0) << "Exhaustive mode requires at least one top blob.";
			checker.CheckGradientExhaustive(&layer, bottom_, top_, 0);
			checker.CheckGradientExhaustive(&layer, bottom_, top_, 1);
		}


	protected:
		void SetUp(){
			vector<int> x_shape{ 5, 2, 2 };
			input_0->Reshape(x_shape);
			Dtype* x_data = input_0->mutable_cpu_data();
			for (int c = 0; c < 5; ++c){
				for (int j = 0; j < 4; j++){
					x_data[c * 4 + j] = c * 0.1 + j * 0.3;
				}
			}
			input_1->Reshape(x_shape);
			Dtype* input_1_data = input_1->mutable_cpu_data();
			for (int c = 0; c < 5; ++c){
				for (int j = 0; j < 4; j++){
					x_data[c * 4 + j] = c * 0.15 + j * 0.2;
				}
			}
			vector<int> label_shape{ 5 };
			input_2->Reshape(label_shape);
			input_3->Reshape(label_shape);
			Dtype* input_2_data = input_2->mutable_cpu_data();
			Dtype* input_3_data = input_3->mutable_cpu_data();
			input_2_data[0] = 0;
			input_2_data[1] = 1;
			input_2_data[2] = 0;
			input_2_data[3] = -1;
			input_2_data[4] = 1;
			input_3_data[0] = 0;
			input_3_data[1] = 1;
			input_3_data[2] = 1;
			input_3_data[3] = 0;
			input_3_data[4] = -1;

			//set bottom && top
			bottom_.push_back(input_0);
			bottom_.push_back(input_1);
			bottom_.push_back(input_2);
			bottom_.push_back(input_3);
			top_.push_back(output_);
			layer_param_.mutable_semi_hinge_loss_param()->set_axis(1);
			layer_param_.mutable_semi_hinge_loss_param()->set_ignore_label(-1);
			layer_param_.mutable_semi_hinge_loss_param()->set_sup_bias(1);
			layer_param_.mutable_semi_hinge_loss_param()->set_unsup_bias(0.5);
			layer_param_.mutable_semi_hinge_loss_param()->set_gamma(0.1);
		}

		Blob<Dtype>* input_0;
		Blob<Dtype>* input_1;
		Blob<Dtype>* input_2;
		Blob<Dtype>* input_3;
		Blob<Dtype>* output_;

		vector<Blob<Dtype>*> bottom_;
		vector<Blob<Dtype>*> top_;

		LayerParameter layer_param_;
	};
}

int main(int argc, char** argv){
	::google::InitGoogleLogging(*argv);
	FLAGS_logtostderr = true;
	caffe::SemiHingeLossLayerTest<float> test;
	test.TestSetUp();
//	test.TestForward(caffe::Caffe::CPU);
//	test.TestForward(caffe::Caffe::GPU);
	test.TestGradients(caffe::Caffe::CPU);
//	test.TestGradients(caffe::Caffe::GPU);
	return 0;
}