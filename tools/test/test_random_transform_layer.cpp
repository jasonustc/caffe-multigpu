#include <cstring>
#include <vector>

#include "cuda_runtime.h"
#include "gtest/gtest.h"
#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/test/test_gradient_check_util.hpp"
#include "caffe/test/test_caffe_main.hpp"
#include "caffe/layers/random_transform_layer.hpp"


namespace caffe{
	template <typename Dtype>
	class RandomTransformTest{
	public:
		RandomTransformTest() : x_(new Blob<Dtype>()), x_trans_(new Blob<Dtype>()){
			this->SetUp();
		}

		~RandomTransformTest(){  delete x_; delete x_trans_; }

		void TestSetUp(){
			shared_ptr<Layer<Dtype>> layer(new RandomTransformLayer<Dtype>(layer_param_));
			layer->SetUp(bottom_, top_);
			CHECK_EQ(top_[0]->shape(0), 1);
			CHECK_EQ(top_[0]->shape(1), 1);
			CHECK_EQ(top_[0]->shape(2), 4);
			CHECK_EQ(top_[0]->shape(3), 4);
		}

		void TestCPUForward(){
			shared_ptr<Layer<Dtype>> layer(new RandomTransformLayer<Dtype>(layer_param_));
			Caffe::set_mode(Caffe::CPU);
			layer->SetUp(bottom_, top_);
			layer->Forward(bottom_, top_);
			const Dtype* x_data = x_->cpu_data();
			const Dtype* top_data = top_[0]->cpu_data();
			//check rotation
			//NOTE: a little bit different with rotation in matlab for small angles
			for (int i = 0; i < 4; i++){
				for (int j = 0; j < 4; j++){
					cout << top_data[i * 4 + j] << "\t";
				}
				cout << "\n";
			}
//			CHECK_EQ(x_data[0], top_data[6]);
//			CHECK_EQ(x_data[1], top_data[3]);
//			CHECK_EQ(x_data[2], top_data[0]);
//			CHECK_EQ(x_data[3], top_data[7]);
//			CHECK_EQ(x_data[4], top_data[4]);
//			CHECK_EQ(x_data[5], top_data[2]);
//			CHECK_EQ(x_data[6], top_data[8]);
//			CHECK_EQ(x_data[7], top_data[5]);
//			CHECK_EQ(x_data[8], top_data[2]);
		}

		void TestGPUForward(){
			shared_ptr<Layer<Dtype>> layer(new RandomTransformLayer<Dtype>(layer_param_));
			layer->SetUp(bottom_, top_);
			Caffe::set_mode(Caffe::GPU);
			layer->Forward(bottom_, top_);
			const Dtype* x_data = x_->cpu_data();
			const Dtype* top_data = top_[0]->cpu_data();
			for (int i = 0; i < 4; i++){
				for (int j = 0; j < 4; j++){
					cout << top_data[i * 4 + j] << "\t";
				}
				cout << "\n";
			}
			//check rotation
//			CHECK_EQ(x_data[0], top_data[6]);
//			CHECK_EQ(x_data[1], top_data[3]);
//			CHECK_EQ(x_data[2], top_data[0]);
//			CHECK_EQ(x_data[3], top_data[7]);
//			CHECK_EQ(x_data[4], top_data[4]);
//			CHECK_EQ(x_data[5], top_data[2]);
//			CHECK_EQ(x_data[6], top_data[8]);
//			CHECK_EQ(x_data[7], top_data[5]);
//			CHECK_EQ(x_data[8], top_data[2]);
		}

		void TestCPUGradients(){
			RandomTransformLayer<Dtype> layer(layer_param_);
			Caffe::set_mode(Caffe::CPU);
			GradientChecker<Dtype> checker(1e-2, 1e-3);
			//because decoding parameters is not correlated with h_enc,
			//so the computed and estimated gradient will be 0
			//checker.CheckGradientExhaustive(&layer, bottom_, top_);
			layer.SetUp(bottom_, top_);
			CHECK_GT(top_.size(), 0) << "Exhaustive mode requires at least one top blob.";
			LOG(INFO) << top_[0]->count();
			checker.CheckGradientExhaustive(&layer, bottom_, top_);
//			for (int i = 0; i < top_[0]->count(); i++){
//				checker.CheckGradientSingle(&layer, bottom_, top_, 0, 0, i);
//			}
		}

		void TestGPUGradients(){
			RandomTransformLayer<Dtype> layer(layer_param_);
			Caffe::set_mode(Caffe::GPU);
			layer.SetUp(bottom_, top_);
			CHECK_GT(top_.size(), 0) << "Exhaustive mode requires at least one top blob.";
			GradientChecker<Dtype> checker(1e-2, 1e-3);
			for (int i = 0; i < top_[0]->count(); i++){
				checker.CheckGradientSingle(&layer, bottom_, top_, 0, 0, i);
			}
		}
		
	protected:
		void SetUp(){
			vector<int> x_shape;
			x_shape.push_back(1);
			x_shape.push_back(1);
			x_shape.push_back(4);
			x_shape.push_back(4);
			x_->Reshape(x_shape);
			FillerParameter filler_param;
			filler_param.set_value(0.1);
			ConstantFiller<Dtype> filler(filler_param);
			filler.Fill(x_);
			Dtype* x_data = x_->mutable_cpu_data();
			for (int i = 0; i < 4; i++){
				for (int j = 0; j < 4; j++){
					x_data[i * 4 + j] = i * 4 + j;
				}
			}
			bottom_.push_back(x_);
			top_.push_back(x_trans_);
			propagate_down_.resize(1, true);

			//set layer param
			//uniform
//			layer_param_.mutable_rand_trans_param()->set_sample_type(RandTransformParameter_SampleType_UNIFORM);
//			layer_param_.mutable_rand_trans_param()->set_start_angle(-90);
//			layer_param_.mutable_rand_trans_param()->set_end_angle(90);
//			layer_param_.mutable_rand_trans_param()->set_dy_prop(0.8);
//			layer_param_.mutable_rand_trans_param()->set_dx_prop(0.1);
//			layer_param_.mutable_rand_trans_param()->set_start_scale(0.3);
//			layer_param_.mutable_rand_trans_param()->set_end_scale(0.7);
			//gaussian
			layer_param_.mutable_rand_trans_param()->set_sample_type(RandTransformParameter_SampleType_GAUSSIAN);
			layer_param_.mutable_rand_trans_param()->set_std_angle(30);
			layer_param_.mutable_rand_trans_param()->set_std_scale(0.1);
			layer_param_.mutable_rand_trans_param()->set_std_dx_prop(0.2);
			layer_param_.mutable_rand_trans_param()->set_std_dy_prop(0.2);
			layer_param_.mutable_rand_trans_param()->set_max_scale(0.3);
			layer_param_.mutable_rand_trans_param()->set_max_shift_prop(0.3);
			layer_param_.mutable_rand_trans_param()->set_alternate(true);
		}

		Blob<Dtype>* x_;

		Blob<Dtype>* x_trans_;

		vector<Blob<Dtype>*> bottom_;
		vector<Blob<Dtype>*> top_;

		vector<bool> propagate_down_;

		LayerParameter layer_param_;
	};
}

int main(int argc, char** argv){
	caffe::RandomTransformTest<float> test;
//	test.TestSetUp();
//	test.TestCPUForward();
	test.TestCPUGradients();
//	test.TestGPUForward();
//	test.TestGPUGradients();
	return 0;
}