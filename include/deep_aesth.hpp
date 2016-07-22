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
		svm_model_file_ = "DCNN_Aesth.xml";
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
			LOG_IF(INFO, (i % 1000 == 0)) << "Loaded " << i << " images";
		}
	}

	void GetFeatFromFolder(string foler_path, string blob_name){
		vector<string> fileList;
		ListFilesInDir(foler_path, fileList);
		for (size_t i = 0; i < fileList.size(); i++){
			LoadImage(fileList[i]);
			GetFeat(blob_name, fileList[i] + ".feat");
			LOG_IF(INFO, (i % 1000 == 0)) << "Loaded " << i << " images";
		}
	}

	void GetFeatFromImage(string img_path, string blob_name){
		LoadImage(img_path);
		GetFeat(blob_name, img_path + ".feat");
	}

	void GetFeat(string blob_name, string feat_path){
		CHECK(net_->has_blob(blob_name));
		net_->Forward();
		Blob<Dtype>* blob = net_->blob_by_name(blob_name).get();
		const Dtype* blob_data = blob->cpu_data();
		std::ofstream out_feat(feat_path.c_str(), std::ios::binary);
		CHECK(out_feat.is_open());
		// fake label
//		out_feat << "+1" << "\t";
		for (int i = 0; i < blob->count(); ++i){
			Dtype data = blob_data[i];
			// only non-zero data will be kept
//			if (abs(data) > 1e-10){
//				out_feat << (i + 1) << ":" << data << "\t";
//			}
			out_feat.write((char*)(&data), sizeof(Dtype));
		}
		out_feat.close();
	}

	void GetFeat(string blob_name){
		this->GetFeat(blob_name, this->buf_file_.c_str());
	}

	float GetScore(){
		FILE* input;
		FILE* output;
		input = fopen(buf_file_.c_str(), "r");
		if (input == NULL)
		{
			fprintf(stderr, "can't open input file %s\n", buf_file_.c_str());
			exit(1);
		}
		output = fopen(pred_file_.c_str(), "w");
		float score = liblinear_predict(input, output, svm_model_file_.c_str(), predict_prob_);
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

protected:
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
	bool predict_prob_ = false;
};