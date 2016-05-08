#include <cstring>
#include <vector>

#include "cuda_runtime.h"
#include "gtest/gtest.h"
#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/test/test_gradient_check_util.hpp"
#include "caffe/test/test_caffe_main.hpp"
#include "caffe/layers/patch_rank_layer.hpp"


namespace caffe{
	template <typename Dtype>
	class PatchRankLayerTest{
	public:
		PatchRankLayerTest() : x_(new Blob<Dtype>()), x_rank_(new Blob<Dtype>()){
			this->SetUp();
		}

		~PatchRankLayerTest(){  delete x_; delete x_rank_; }

		void TestSetUp(){
			shared_ptr<Layer<Dtype>> layer(new PatchRankLayer<Dtype>(layer_param_));
			layer->SetUp(bottom_, top_);
			CHECK(top_[0]->shape() == bottom_[0]->shape());
		}

		void TestForward(Caffe::Brew mode){
			shared_ptr<Layer<Dtype>> layer(new PatchRankLayer<Dtype>(layer_param_));
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
			PatchRankLayer<Dtype> layer(layer_param_);
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
				CHECK_EQ(bottom_data[i], bottom_diff[i]) <<"index: " << i 
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
			x_shape.push_back(9);
			x_shape.push_back(10);
			x_->Reshape(x_shape);
			FillerParameter filler_param;
			filler_param.set_value(0.1);
			ConstantFiller<Dtype> filler(filler_param);
			filler.Fill(x_);
			Dtype* x_data = x_->mutable_cpu_data();
			for (int c = 0; c < 6; ++c){
				for (int i = 0; i < 9; i++){
					for (int j = 0; j < 10; j++){
						x_data[c * 90 + i * 10 + j] = j * 10 + i;
					}
				}
			}
			bottom_.push_back(x_);
			top_.push_back(x_rank_);
			propagate_down_.resize(1, true);

			//set layer param
			layer_param_.mutable_patch_rank_param()->set_block_num(2);
			layer_param_.mutable_patch_rank_param()->set_energy_type(
				PatchRankParameter_EnergyType_L1);
			layer_param_.mutable_patch_rank_param()->set_pyramid_height(2);
		}

		Blob<Dtype>* x_;

		Blob<Dtype>* x_rank_;

		vector<Blob<Dtype>*> bottom_;
		vector<Blob<Dtype>*> top_;

		vector<bool> propagate_down_;

		LayerParameter layer_param_;
	};
}

int main(int argc, char** argv){
	::google::InitGoogleLogging(*argv);
	FLAGS_logtostderr = true;
	caffe::PatchRankLayerTest<float> test;
//	test.TestSetUp();
//	test.TestForward(caffe::Caffe::CPU);
//	test.TestGradients(caffe::Caffe::CPU);
	test.TestForward(caffe::Caffe::GPU);
	test.TestGradients(caffe::Caffe::GPU);
	return 0;
}