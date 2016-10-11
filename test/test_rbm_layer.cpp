#include <cstring>
#include <vector>

#include "cuda_runtime.h"
#include "gtest/gtest.h"
#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/test/test_gradient_check_util.hpp"
#include "caffe/test/test_caffe_main.hpp"
#include "caffe/layers/rbm_layer.hpp"


namespace caffe{
	template <typename Dtype>
	class RBMLayerTest{
	public:
		RBMLayerTest() : x_(new Blob<Dtype>()), out_(new Blob<Dtype>()), loss_(new Blob<Dtype>()){
			this->SetUp();
		}

		void TestSetUp(){
			shared_ptr<Layer<Dtype>> layer(new RBMLayer<Dtype>(layer_param_));
			layer->SetUp(bottom_, top_);
			LOG(INFO) << bottom_[0]->shape_string();
			CHECK_EQ(top_[0]->shape(0), 1);
			CHECK_EQ(top_[0]->shape(1), 3);
		}

		void TestForward(Caffe::Brew mode){
			shared_ptr<Layer<Dtype>> layer(new RBMLayer<Dtype>(layer_param_));
			Caffe::set_mode(mode);
			LOG(INFO) << bottom_[0]->shape_string();
			layer->SetUp(bottom_, top_);
			layer->Forward(bottom_, top_);
			layer->Forward(bottom_, top_);
			layer->Forward(bottom_, top_);
			layer->Forward(bottom_, top_);
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
			RBMLayer<Dtype> layer(layer_param_);
			CHECK_GT(top_.size(), 0) << "Exhaustive mode requires at least one top blob.";
			Caffe::set_mode(mode);
			layer.SetUp(bottom_, top_);
			layer.Forward(bottom_, top_);
			layer.Backward(top_, vector<bool>(1, true), bottom_);
			Dtype v0 = layer.pos_v_->cpu_data()[0];
			Dtype h0 = layer.pos_h_->cpu_data()[0];
			Dtype vk = layer.neg_v_->cpu_data()[0];
			Dtype hk = layer.neg_h_->cpu_data()[0];
			Dtype dw_t = v0 * h0 - vk * hk;
			EXPECT_NEAR(dw_t, layer.blobs()[0]->cpu_diff()[0], 1e-3);
		}


	protected:
		void SetUp(){
			vector<int> x_shape{2, 20};
			x_->Reshape(x_shape);
			Dtype* x_data = x_->mutable_cpu_data();
			for (int c = 0; c < 2; ++c){
				for (int j = 0; j < 20; j++){
					x_data[c * 2 + j] = c * 0.1 + j * 0.3;
				}
			}

			//set bottom && top
			bottom_.push_back(x_);
			top_.push_back(out_);
			top_.push_back(loss_);
			propagate_down_.resize(1, true);

			// set layer_param_
			layer_param_.mutable_inner_product_param()->set_num_output(3);
			layer_param_.mutable_inner_product_param()->mutable_weight_filler()->set_type("gaussian");
			layer_param_.mutable_inner_product_param()->mutable_weight_filler()->set_std(0.1);
			layer_param_.mutable_inner_product_param()->mutable_bias_filler()->set_type("constant");
			layer_param_.mutable_inner_product_param()->mutable_bias_filler()->set_value(0.);
			layer_param_.mutable_inner_product_param()->set_axis(1);
			layer_param_.mutable_rbm_param()->set_num_iteration(2);
			layer_param_.mutable_rbm_param()->set_learn_by_cd(true);
			layer_param_.mutable_rbm_param()->set_learn_by_top(true);
			layer_param_.mutable_sampling_param()->set_sample_type(SamplingParameter_SampleType_BERNOULLI);
			layer_param_.mutable_rbm_param()->set_block_start(3);
			layer_param_.mutable_rbm_param()->set_block_end(10);
			layer_param_.mutable_rbm_param()->set_random_block(true);
		}

		Blob<Dtype>* x_;
		Blob<Dtype>* out_;
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
	caffe::RBMLayerTest<float> test;
//	test.TestSetUp();
	test.TestForward(caffe::Caffe::CPU);
	test.TestGradients(caffe::Caffe::CPU);
	test.TestForward(caffe::Caffe::GPU);
	test.TestGradients(caffe::Caffe::GPU);
	return 0;
}