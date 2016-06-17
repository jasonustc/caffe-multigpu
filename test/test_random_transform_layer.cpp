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
#include "opencv2/opencv.hpp"


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
//			CHECK_EQ(top_[0]->shape(0), 1);
//			CHECK_EQ(top_[0]->shape(3), 1);
//			CHECK_EQ(top_[0]->shape(2), 4);
//			CHECK_EQ(top_[0]->shape(3), 4);
		}

		void TestForward(caffe::Caffe::Brew mode){
			shared_ptr<Layer<Dtype>> layer(new RandomTransformLayer<Dtype>(layer_param_));
			Caffe::set_mode(mode);
			layer->SetUp(bottom_, top_);
			layer->Forward(bottom_, top_);
			cv::Mat transImg;
			x_trans_->ToMat(transImg, CV_8U);
			cv::imshow("transImg", transImg);
			cv::imwrite("transImg1.jpg", transImg);
			cv::waitKey();
			layer->Forward(bottom_, top_);
			x_trans_->ToMat(transImg, CV_8U);
			cv::imshow("transImg", transImg);
			cv::imwrite("transImg2.jpg", transImg);
			cv::waitKey();
			layer->Forward(bottom_, top_);
			x_trans_->ToMat(transImg, CV_8U);
			cv::imshow("transImg", transImg);
			cv::imwrite("transImg3.jpg", transImg);
			cv::waitKey();
			layer->Forward(bottom_, top_);
			x_trans_->ToMat(transImg, CV_8U);
			cv::imshow("transImg", transImg);
			cv::imwrite("transImg4.jpg", transImg);
			cv::waitKey();
			layer->Forward(bottom_, top_);
			x_trans_->ToMat(transImg, CV_8U);
			cv::imshow("transImg", transImg);
			cv::imwrite("transImg5.jpg", transImg);
			cv::waitKey();
			layer->Forward(bottom_, top_);
			x_trans_->ToMat(transImg, CV_8U);
			cv::imshow("transImg", transImg);
			cv::imwrite("transImg6.jpg", transImg);
			cv::waitKey();
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
			cv::Mat img = cv::imread("ILSVRC2012_val_00000001.JPEG", true);
			x_->FromMat(img);
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
//			layer_param_.mutable_rand_trans_param()->set_sample_type(RandTransformParameter_SampleType_GAUSSIAN);
//			layer_param_.mutable_rand_trans_param()->set_std_angle(30);
//			layer_param_.mutable_rand_trans_param()->set_std_scale(0.1);
//			layer_param_.mutable_rand_trans_param()->set_std_dx_prop(0.2);
//			layer_param_.mutable_rand_trans_param()->set_std_dy_prop(0.2);
//			layer_param_.mutable_rand_trans_param()->set_max_scale(0.3);
//			layer_param_.mutable_rand_trans_param()->set_max_shift_prop(0.3);
//			layer_param_.mutable_rand_trans_param()->set_alternate(true);
			//totally random
			layer_param_.mutable_rand_trans_param()->set_sample_type(RandTransformParameter_SampleType_UNIFORM);
			layer_param_.mutable_rand_trans_param()->set_total_random(true);
			layer_param_.mutable_rand_trans_param()->set_rand_param1(-0.3);
			layer_param_.mutable_rand_trans_param()->set_rand_param2(0.3);
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
	::google::InitGoogleLogging(argv[0]);
	FLAGS_alsologtostderr = true;
	::google::SetStderrLogging(0);
	caffe::RandomTransformTest<float> test;
	test.TestSetUp();
	test.TestForward(caffe::Caffe::GPU);
//	test.TestForward(caffe::Caffe::CPU);
	return 0;
}