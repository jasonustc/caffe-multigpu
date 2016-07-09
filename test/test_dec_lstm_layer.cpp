#include <cstring>
#include <vector>

#include "cuda_runtime.h"
#include "gtest/gtest.h"
#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/test/test_gradient_check_util.hpp"
#include "caffe/test/test_caffe_main.hpp"
#include "caffe/layers/dec_lstm_layer.hpp"


namespace caffe{
	template <typename Dtype>
	class DLSTMLayerTest{
	public:
		DLSTMLayerTest() : x_(new Blob<Dtype>()), x_rank_(new Blob<Dtype>()){
			this->SetUp();
		}

		~DLSTMLayerTest(){ delete x_; delete x_rank_; }

		void TestSetUp(){
			shared_ptr<Layer<Dtype>> layer(new DLSTMLayer<Dtype>(layer_param_));
			layer->SetUp(bottom_, top_);
			CHECK(top_[0]->shape() == bottom_[0]->shape());
		}

		void TestForward(Caffe::Brew mode){
			shared_ptr<Layer<Dtype>> layer(new DLSTMLayer<Dtype>(layer_param_));
			Caffe::set_mode(mode);
			layer->SetUp(bottom_, top_);
			layer->Forward(bottom_, top_);
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
			DLSTMLayer<Dtype> layer(layer_param_);
			Caffe::set_mode(mode);
			GradientChecker<Dtype> checker(0.01, 0.001);
			//because decoding parameters is not correlated with h_enc,
			//so the computed and estimated gradient will be 0
			//checker.CheckGradientExhaustive(&layer, bottom_, top_);
			layer.SetUp(bottom_, top_);
			Dtype* bottom_data = bottom_[0]->mutable_cpu_data();
			Dtype* bottom_diff = bottom_[0]->mutable_cpu_diff();
			layer.Forward(bottom_, top_);
			Dtype* top_diff = top_[0]->mutable_cpu_diff();
			Dtype* top_data = top_[0]->mutable_cpu_data();
			caffe_copy<Dtype>(top_[0]->count(), top_data, top_diff);
			top_[0]->ToTxt("top", true);
			layer.Backward(top_, vector<bool>(1, true), bottom_);
			bottom_[0]->ToTxt("bottom", true);
			for (int i = 0; i < bottom_[0]->count(); ++i){
				CHECK_EQ(bottom_data[i], bottom_diff[i]) << "index: " << i
					<< "\tdata_i: "
					<< bottom_data[i] << " diff_i: " << bottom_diff[i];
			}
			CHECK_GT(top_.size(), 0) << "Exhaustive mode requires at least one top blob.";
			checker.CheckGradientExhaustive(&layer, bottom_, top_);
		}


	protected:
		void SetUp(){
			vector<int> x_shape;
			x_shape.push_back(2);
			x_shape.push_back(3);
			x_shape.push_back(2);
			x_shape.push_back(2);
			x_->Reshape(x_shape);
			Dtype* x_data = x_->mutable_cpu_data();
			for (int c = 0; c < 3; ++c){
				for (int i = 0; i < 2; i++){
					for (int j = 0; j < 2; j++){
						x_data[c * 4 + i * 2 + j] = j * 2 + i;
					}
				}
			}
			for (int c = 0; c < 3; ++c){
				for (int i = 0; i < 2; i++){
					for (int j = 0; j < 2; j++){
						x_data[12 + c * 4 + i * 2 + j] = i * 2 + j;
					}
				}
			}

			//set bottom && top
			bottom_.push_back(c0_);
			bottom_.push_back(h0_);
			bottom_.push_back(x_);
			top_.push_back(y_);
			propagate_down_.resize(1, true);

			//set layer param
		}

		Blob<Dtype>* c0_;
		Blob<Dtype>* h0_;
		Blob<Dtype>* x_;
		Blob<Dtype>* y_;

		vector<Blob<Dtype>*> bottom_;
		vector<Blob<Dtype>*> top_;

		vector<bool> propagate_down_;

		LayerParameter layer_param_;
	};
}

int main(int argc, char** argv){
	::google::InitGoogleLogging(*argv);
	FLAGS_logtostderr = true;
	caffe::DLSTMLayerTest<float> test;
	test.TestSetUp();
	test.TestForward(caffe::Caffe::CPU);
	test.TestGradients(caffe::Caffe::CPU);
	test.TestForward(caffe::Caffe::GPU);
	test.TestGradients(caffe::Caffe::GPU);
	return 0;
}