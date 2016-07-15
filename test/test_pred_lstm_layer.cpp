#include <cstring>
#include <vector>

#include "cuda_runtime.h"
#include "gtest/gtest.h"
#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/test/test_gradient_check_util.hpp"
#include "caffe/test/test_caffe_main.hpp"
#include "caffe/layers/pred_lstm_layer.hpp"
#include "caffe/layers/dec_lstm_unit_layer.hpp"


namespace caffe{
	template <typename Dtype>
	class PredLSTMLayerTest{
	public:
		PredLSTMLayerTest() : h_(new Blob<Dtype>()), y_(new Blob<Dtype>()){
			this->SetUp();
		}

		~PredLSTMLayerTest(){ delete h_;  delete y_; }

		void TestSetUp(){
			shared_ptr<Layer<Dtype>> layer(new PredLSTMLayer<Dtype>(layer_param_));
			layer->SetUp(bottom_, top_);
			CHECK_EQ(top_[0]->shape(0), 6);
			CHECK_EQ(top_[0]->shape(1), 1);
			CHECK_EQ(top_[0]->shape(2), 4);
		}

		void TestForward(Caffe::Brew mode){
			shared_ptr<Layer<Dtype>> layer(new PredLSTMLayer<Dtype>(layer_param_));
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

	protected:
		void SetUp(){
			vector<int> h_shape{2, 1, 2};
			h_->Reshape(h_shape);
			Dtype* h_data = h_->mutable_cpu_data();
			for (int c = 0; c < 2; ++c){
				for (int j = 0; j < 2; j++){
					h_data[c * 2 + j] = c * 0.1 + j * 0.3;
				}
			}

			//set bottom && top
			bottom_.push_back(h_);
			top_.push_back(y_);
			propagate_down_.resize(1, true);

			// set layer_param_
			layer_param_.mutable_inner_product_param()->set_num_output(3);
			layer_param_.mutable_inner_product_param()->mutable_weight_filler()->set_type("gaussian");
			layer_param_.mutable_inner_product_param()->mutable_weight_filler()->set_std(0.1);
			layer_param_.mutable_inner_product_param()->mutable_bias_filler()->set_type("constant");
			layer_param_.mutable_inner_product_param()->mutable_bias_filler()->set_value(0.);
			layer_param_.mutable_recurrent_param()->set_output_dim(4);
			layer_param_.mutable_recurrent_param()->set_pred_length(3);
		}

		Blob<Dtype>* h_;
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
	caffe::PredLSTMLayerTest<float> test;
	test.TestSetUp();
	test.TestForward(caffe::Caffe::CPU);
	test.TestForward(caffe::Caffe::GPU);
	return 0;
}