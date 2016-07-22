#ifndef CAFFE_FEAT_EXTRACTOR_HPP_
#define CAFFE_FEAT_EXTRACTOE_HPP_

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
#include "caffe/util/file_proc_util.h"

using namespace cv;

namespace caffe{
	/*
	 * @brief The net must include an input layer named as "data" with a type "Input"
	 * For the input layer:
	 *  - net input shapes should be set in input_param
	 *  - reshape parameters should be set in image_data_param
	 *  - transformation parameters should be set in transform_param
	 */
	template <typename Dtype>
	class FeatExtractor {
	public:
		FeatExtractor(string net_file, string model_file, string feat_path,
			Caffe::Brew mode = Caffe::CPU){
			Init(net_file, model_file, mode);
			feat_path_ = feat_path;
			single_file_ = true;
		}

		FeatExtractor(string net_file, string model_file, Caffe::Brew mode = Caffe::CPU){
			Init(net_file, model_file, mode);
			single_file_ = false;
		}

		void GetFeatFromIndexFile(string index_path, string blob_name){
			std::ifstream in_path(index_path.c_str());
			CHECK(in_path.is_open());
			in_path.close();
			vector<string> fileList;
			LoadPathFromFile(index_path, fileList);
			GetFeatFromList(fileList, blob_name);
		}

		void GetFeatFromFolder(string foler_path, string blob_name){
			vector<string> fileList;
			ListFilesInDir(foler_path, fileList);
			GetFeatFromList(fileList, blob_name);
		}

		void GetFeatFromImage(string img_path, string blob_name){
			batch_size_ = 1;
			input_shape_change_ = true;
			LoadImage(img_path, net_input_);
			GetFeat(blob_name, vector<string>(1, img_path));
		}

	private:

		void Init(string net_file, string model_file, Caffe::Brew mode = Caffe::CPU){
			net_ = new Net<Dtype>(net_file, caffe::TEST);
			Caffe::set_mode(mode);
			net_->CopyTrainedLayersFrom(model_file);
			CHECK(net_->has_layer("data"));
			Layer<Dtype>* input_layer = net_->layer_by_name("data").get();
			LayerParameter layer_param = input_layer->layer_param();
			InputParameter input_param = layer_param.input_param();
			batch_size_ = input_param.shape(0).dim(0);
			channels_ = input_param.shape(0).dim(1);
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
			// temp blob for a single image
			temp_ = new Blob<Dtype>();
			// setup memory for net input mannually
			if (crop_size_){
				net_input_->Reshape(batch_size_, channels_, crop_size_, crop_size_);
				temp_->Reshape(1, channels_, crop_size_, crop_size_);
				input_shape_change_ = false;
			}
			else if (new_height_ && new_width_){
				net_input_->Reshape(batch_size_, channels_, new_height_, new_width_);
				temp_->Reshape(1, channels_, new_height_, new_width_);
				input_shape_change_ = false;
			}
			else{
				input_shape_change_ = true;
				CHECK_EQ(batch_size_, 1) << "Since the input shape changes for every sample, "
					<< "batch_size should be set to 1";
			}
		}

		void LoadImage(string image_path, Blob<Dtype>* blob, const int offset = 0){
			cv::Mat cv_img;
			// reshape if needed
			cv_img = ReadImageToCVMat(root_folder_ + image_path, new_height_,
				new_width_, is_color_);
			CHECK(cv_img.data) << "Could not load " << image_path;
			if (input_shape_change_){
				// reshape to deal with change of input image shapes
				const int height = crop_size_ > 0 ? crop_size_ : cv_img.rows;
				const int width = crop_size_ > 0 ? crop_size_ : cv_img.cols;
				net_input_->Reshape(batch_size_, channels_, height, width);
				temp_->Reshape(1, channels_, height, width);
			}
			// substract mean and crop if needed
			this->data_transformer_->Transform(cv_img, temp_);
			if (Caffe::mode() == Caffe::GPU){
				Dtype* input_data = net_input_->mutable_gpu_data() + net_input_->offset(offset);
				caffe_copy(temp_->count(), temp_->gpu_data(), input_data);
			}
			else{
				Dtype* input_data = net_input_->mutable_cpu_data() + net_input_->offset(offset);
				caffe_copy(temp_->count(), temp_->cpu_data(), input_data);
			}
		}

		void LoadBatch(vector<string>& batch_path){
			for (size_t i = 0; i < batch_path.size(); ++i){
				string img_path = batch_path[i];
				LoadImage(img_path, net_input_, i);
			}
		}

		void GetFeatFromList(vector<string> fileList, string blob_name){
			const int N = fileList.size() / batch_size_;
			const int R = fileList.size() % batch_size_;
			vector<string> batch_path;
			batch_path.resize(batch_size_);
			for (int n = 0; n < N; n++){
				for (int b = 0; b < batch_size_; ++b){
					batch_path[b] = fileList[n * batch_size_ + b];
				}
				LoadBatch(batch_path);
				if (single_file_){
					GetFeat(blob_name, feat_path_);
				}
				else{
					GetFeat(blob_name, batch_path);
				}
				LOG_IF(INFO, ((n + 1) % 100 == 0)) << "Loaded " << n << " batches";
			}
			// last batch
			for (int b = 0; b < R; b++){
				batch_path[b] = fileList[N * batch_size_ + b];
				LoadBatch(batch_path);
				if (single_file_){
					GetFeat(blob_name, feat_path_);
				}
				else{
					GetFeat(blob_name, batch_path);
				}
			}
			LOG(INFO) << "Feats of " << fileList.size() << " images extracted";
		}

		// all in a single file
		void GetFeat(string blob_name, string feat_path){
			CHECK(net_->has_blob(blob_name));
			net_->Forward();
			Blob<Dtype>* blob = net_->blob_by_name(blob_name).get();
			const Dtype* blob_data = blob->cpu_data();
			std::ofstream out_feat;
			out_feat.open(feat_path.c_str(), std::ios::binary | std::ios::app);
			CHECK(out_feat.is_open());
			out_feat.write((char*)(blob_data), sizeof(Dtype) * blob->count());
			out_feat.close();
		}

		// each image a single file
		void GetFeat(string blob_name, vector<string> paths){
			CHECK(net_->has_blob(blob_name));
			CHECK(net_input_->shape(0), paths.size());
			net_->Forward();
			Blob<Dtype>* blob = net_->blob_by_name(blob_name).get();
			const Dtype* blob_data = blob->cpu_data();
			const Dtype* first = blob->cpu_data();
			const Dtype* second = blob->cpu_data() + blob->offset(1);
			const int img_count = blob->count(1);
			for (size_t i = 0; i < blob->shape(0); ++i){
				std::ofstream out_feat;
				string feat_path = paths[i] + ".feat";
				out_feat.open(feat_path.c_str(), std::ios::binary);
				CHECK(out_feat.is_open());
				out_feat.write((char*)(blob_data), sizeof(Dtype)* img_count);
				out_feat.close();
				blob_data += blob->offset(1);
			}
		}

	private:
		Net<Dtype>* net_;
		Blob<Dtype>* net_input_;
		shared_ptr<DataTransformer<Dtype> > data_transformer_;
		int new_height_;
		int new_width_;
		int crop_size_;
		int channels_;
		string root_folder_;
		string feat_path_;
		// if output all the features into one single file
		bool single_file_;
		bool is_color_;
		int batch_size_;
		Blob<Dtype>* temp_;
		bool input_shape_change_;
	};
}
#endif