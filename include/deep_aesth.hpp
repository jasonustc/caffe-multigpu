#include <vector>
#include <utility>
#include <fstream>
#include <iostream>
#include <io.h>

#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/common.hpp"
#include "caffe/blob.hpp"
#include "caffe/net.hpp"
#include "caffe/util/io.hpp"
#include "opencv2/opencv.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/layers/input_layer.hpp"
#include "../src/liblinear/predict.h"
#include "deep_aesth_util.hpp"

using namespace caffe;
using namespace cv;

template <typename Dtype>
class DeepAesth {
public:
	DeepAesth(string& net_file, string& model_file){
		net_ = new Net<Dtype>(net_file, caffe::TEST);
		Caffe::set_mode(Caffe::CPU);
		net_->CopyTrainedLayersFrom(model_file);
		CHECK(net_->has_layer("data"));
		Layer<Dtype>* input_layer = net_->layer_by_name("data").get();
		LayerParameter layer_param = input_layer->layer_param();
		TransformationParameter trans_param = layer_param.transform_param();
		data_transformer_.reset(new DataTransformer<Dtype>(trans_param, caffe::TEST));
		data_transformer_->InitRand();
		crop_size_ = trans_param.crop_size();
		ImageDataParameter image_data_param = layer_param.image_data_param();
		new_height_ = image_data_param.new_height();
		new_width_ = image_data_param.new_width();
		root_folder_ = image_data_param.root_folder();
		is_color_ = image_data_param.is_color();
		CHECK(net_->has_blob("data"));
		net_input_ = net_->blob_by_name("data").get();
		buf_file_ = "buf";
		pred_file_ = "score";
		svm_model_file_ = FLAGS_svm_model;
		this->SetL2Norm(FLAGS_l2_norm);
		this->SetSqrt(FLAGS_sqrt);
	}

	void LoadImage(string image_path){
		cv::Mat cv_img = ReadImageToCVMat(root_folder_ + image_path, 0, 
			0, is_color_);
		CHECK(cv_img.data) << "Could not load " << image_path;
		vector<int> input_shape = {1, 3, crop_size_, crop_size_};
		Blob<Dtype>* temp = new Blob<Dtype>(input_shape);
		// 6 blocks 
		input_shape[0] = 6;
		net_input_->Reshape(input_shape);
		Dtype* net_input_data = net_input_->mutable_cpu_data();
		Image2Blocks(cv_img);
		for (int i = 0; i < 6; ++i){
			cv::Mat image_resize;
			cv::resize(image_blocks_[i], image_resize, cv::Size(new_width_, new_height_));
			// subtract mean and crop
			this->data_transformer_->Transform(image_resize, temp);
			caffe_copy(temp->count(), temp->cpu_data(), net_input_data);
			net_input_data += temp->count();
		}
	}

	void LoadImage(cv::Mat& cv_img){
		CHECK(cv_img.data) << "Could not load " << image_path;
		vector<int> input_shape = {1, 3, crop_size_, crop_size_};
		Blob<Dtype>* temp = new Blob<Dtype>(input_shape);
		// 6 blocks 
		input_shape[0] = 6;
		net_input_->Reshape(input_shape);
		Dtype* net_input_data = net_input_->mutable_cpu_data();
		Image2Blocks(cv_img);
		for (int i = 0; i < 6; ++i){
			cv::Mat image_resize;
			cv::resize(image_blocks_[i], image_resize, cv::Size(new_width_, new_height_));
			// subtract mean and crop
			this->data_transformer_->Transform(image_resize, temp);
			caffe_copy(temp->count(), temp->cpu_data(), net_input_data);
			net_input_data += temp->count();
		}
	}

	void GetFeatFromIndexFile(string index_path, string blob_name){
		std::ifstream in_path(index_path.c_str());
		CHECK(in_path.is_open());
		in_path.close();
		vector<string> fileList;
		LoadPathFromFile(index_path, fileList);
		for (size_t i = 0; i < fileList.size(); i++){
			LoadImage(fileList[i]);
			GetFeat(blob_name, fileList[i] + ".feat");
			LOG_IF(INFO, ((i + 1) % 1000 == 0)) << "Loaded " << i << " images";
		}
	}

	void GetFeatFromFolder(string foler_path, string blob_name){
		vector<string> fileList;
		ListFilesInDir(foler_path, fileList);
		for (size_t i = 0; i < fileList.size(); i++){
			LoadImage(fileList[i]);
			GetFeat(blob_name, fileList[i] + ".feat");
			LOG_IF(INFO, ((i + 1) % 1000 == 0)) << "Loaded " << i << " images";
		}
	}

	void GetFeatFromImage(string img_path, string blob_name){
		LoadImage(img_path);
		GetFeat(blob_name, img_path + ".feat");
	}

	void GetFeat(string blob_name, string feat_path, const bool binary = true){
		CHECK(net_->has_blob(blob_name));
		net_->Forward();
		Blob<Dtype>* blob = net_->blob_by_name(blob_name).get();
		const Dtype* blob_data = blob->cpu_data();
		if (sqrt_){
			caffe_powx<Dtype>(blob->count(), blob->mutable_cpu_data(),
				Dtype(0.5), blob->mutable_cpu_data());
		}
		if (l2_norm_){
			this->L2Normalize(blob->count(), blob->mutable_cpu_data());
		}
		if (binary){
			std::ofstream out_feat(feat_path.c_str(), std::ios::binary);
			CHECK(out_feat.is_open());
			out_feat.write((char*)(blob_data), sizeof(Dtype)* blob->count());
			out_feat.close();
		}
		else{
			std::ofstream out_feat(feat_path.c_str());
			CHECK(out_feat.is_open());
			// fake label
			out_feat << "+1";
			for (int i = 0; i < blob->count(); ++i){
				if (abs(blob_data[i]) > 1e-10){
					out_feat << "\t" << i + 1 << ":" << blob_data[i];
				}
			}
			out_feat << "\n";
			out_feat.close();
		}
	}

	void GetFeat(string blob_name){
		this->GetFeat(blob_name, this->buf_file_.c_str(), false);
	}

	float GetScore(){
		FILE* input;
		input = fopen(buf_file_.c_str(), "r");
		if (input == NULL)
		{
			fprintf(stderr, "can't open input file %s\n", buf_file_.c_str());
			exit(1);
		}
		float score = liblinear_predict(input, svm_model_file_.c_str(), predict_prob_);
		return score;
	}

	void Image2Blocks(cv::Mat& image){
		CHECK(image.rows > 2 && image.cols > 2);
		image_blocks_.resize(6);
		int blockWidth = image.cols / 2;
		int blockHeight = image.rows / 2;
		image_blocks_[0] = image;
		image_blocks_[1] = image(Range(0, blockHeight), Range(0, blockWidth));
		image_blocks_[2] = image(Range(blockHeight, image.rows), Range(0, blockWidth));
		image_blocks_[3] = image(Range(0, blockHeight), Range(blockWidth, image.cols));
		image_blocks_[4] = image(Range(blockHeight, image.rows), Range(blockWidth, image.cols));
		image_blocks_[5] = image(Range(image.rows / 4, image.rows * 3 / 4),
			Range(image.cols / 4, image.cols * 3 / 4));
	}

	void SetSqrt(const bool sqrt){
		this->sqrt_ = sqrt;
		LOG_IF(INFO, sqrt) << "will do square root of features";
		LOG_IF(INFO, !sqrt) << "will not do square root of features";
	}
	void SetL2Norm(const bool l2_norm){
		this->l2_norm_ = l2_norm;
		LOG_IF(INFO, l2_norm) << "will do l2 normalization of features";
		LOG_IF(INFO, !l2_norm) << "will not do l2 normalization of features";
	}

protected:

	/*
	* x_i = x_i / ||x||_2
	*/
	template<typename Dtype>
	void L2Normalize(const int dim, Dtype* feat_data){
		Dtype l2_norm = caffe_cpu_dot<Dtype>(dim, feat_data, feat_data) + 1e-9;
		l2_norm = sqrt(l2_norm);
		caffe_scal<Dtype>(dim, Dtype(1. / l2_norm), feat_data);
	}

	Net<Dtype>* net_;
	Blob<Dtype>* net_input_;
	shared_ptr<DataTransformer<Dtype> > data_transformer_;
	vector<Mat> image_blocks_;
	string buf_file_;
	string pred_file_;
	string svm_model_file_;
	int new_height_;
	int new_width_;
	int crop_size_;
	string root_folder_;
	bool is_color_;
	bool predict_prob_ = true;
	bool sqrt_ = false;
	bool l2_norm_ = false;
};