#include <cstring>
#include <vector>

#include "cuda_runtime.h"
#include "gtest/gtest.h"
#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/test/test_gradient_check_util.hpp"
#include "caffe/test/test_caffe_main.hpp"
#include "caffe/layers/softmax_loss_layer.hpp"


namespace caffe{
	template <typename Dtype>
	class SoftmaxWithLossLayerTest{
	public:
		SoftmaxWithLossLayerTest() : x_(new Blob<Dtype>()), label_(new Blob<Dtype>()), loss_(new Blob<Dtype>()){
			this->SetUp();
		}

		~SoftmaxWithLossLayerTest(){ delete x_; delete label_; }

		void TestSetUp(){
			shared_ptr<Layer<Dtype>> layer(new SoftmaxWithLossLayer<Dtype>(layer_param_));
			layer->SetUp(bottom_, top_);
		}

		void TestForward(Caffe::Brew mode){
			shared_ptr<Layer<Dtype>> layer(new SoftmaxWithLossLayer<Dtype>(layer_param_));
			Caffe::set_mode(mode);
			layer->SetUp(bottom_, top_);
			layer->Forward(bottom_, top_);
			const Dtype* x_data = bottom_[0]->cpu_data();
			// label weight 2
			Dtype loss_0 = -log(exp(x_data[1]) / (exp(x_data[0]) + exp(x_data[1]) + exp(x_data[2]))) * 2;
			x_data += 3;
			// label weight 3
			Dtype loss_1 = -log(exp(x_data[2]) / (exp(x_data[0]) + exp(x_data[1]) + exp(x_data[2]))) * 3;
			Dtype loss = (loss_0 + loss_1) / 2;
			EXPECT_NEAR(loss, top_[0]->cpu_data()[0], 0.001);
			if (mode == Caffe::CPU){
				bottom_[0]->ToTxt("bottom_cpu");
				top_[0]->ToTxt("top_cpu");
			}
			else{
				top_[0]->ToTxt("top_gpu");
				bottom_[0]->ToTxt("bottom_gpu");
			}
		}


		void TestGradients(Caffe::Brew mode){
			SoftmaxWithLossLayer<Dtype> layer(layer_param_);
			Caffe::set_mode(mode);
			GradientChecker<Dtype> checker(0.01, 0.001);
			//because decoding parameters is not correlated with h_enc,
			//so the computed and estimated gradient will be 0
			//checker.CheckGradientExhaustive(&layer, bottom_, top_);
			CHECK_GT(top_.size(), 0) << "Exhaustive mode requires at least one top blob.";
//			checker.CheckGradientExhaustive(&layer, bottom_, top_);
			checker.CheckGradientExhaustive(&layer, bottom_, top_, 0);
		}

		
	protected:
		void SetUp(){
			vector<int> x_shape;
			x_shape.push_back(2);
			x_shape.push_back(3);
			x_->Reshape(x_shape);
			vector<int> label_shape{ 2 };
			label_->Reshape(label_shape);
			Dtype* x_data = x_->mutable_cpu_data();
			Dtype* label_data = label_->mutable_cpu_data();
			for (int i = 0; i < 2; i++){
				for (int c = 0; c < 3; ++c){
					x_data[i * 2 + c] = i * 0.5 + c;
				}
			}
			label_data[0] = 1;
			label_data[1] = 2;
			bottom_.push_back(x_);
			bottom_.push_back(label_);
			top_.push_back(loss_);
			propagate_down_.resize(1, true);

			//set layer param
			layer_param_.mutable_loss_param()->set_label_weight_file("label_weight.txt");
			layer_param_.mutable_softmax_param()->set_axis(1);
		}

		Blob<Dtype>* x_;

		Blob<Dtype>* label_;

		Blob<Dtype>* loss_;

		vector<Blob<Dtype>*> bottom_;
		vector<Blob<Dtype>*> top_;

		vector<bool> propagate_down_;

		LayerParameter layer_param_;
	};
}

int main(int argc, char** argv){
	::google::InitGoogleLogging(*argv);
	FLAGS_logtostderr = true;
	caffe::SoftmaxWithLossLayerTest<float> test;
//	test.TestSetUp();
//	test.TestForward(caffe::Caffe::CPU);
	test.TestGradients(caffe::Caffe::CPU);
//	test.TestForward(caffe::Caffe::GPU);
	test.TestGradients(caffe::Caffe::GPU);
	return 0;
}