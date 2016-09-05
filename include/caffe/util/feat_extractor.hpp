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
#include "caffe/util/math_functions.hpp"

using namespace caffe;

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
			if (mode == Caffe::GPU){
				LOG(INFO) << "using GPU";
				// just use default GPU 0
				Caffe::SetDevice(0);
			}
			else{
				LOG(INFO) << "using CPU";
			}
			Init(net_file, model_file, mode);
			feat_path_ = feat_path;
			single_file_ = true;
		}

		FeatExtractor(string net_file, string model_file, Caffe::Brew mode = Caffe::CPU){
			if (mode == Caffe::GPU){
				// just use default GPU 0
				Caffe::SetDevice(0);
				LOG(INFO) << "using GPU";
			}
			else{
				LOG(INFO) << "using CPU";
			}
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
				batch_size_ = 1;
				LOG(INFO) << "Since the input shape changes for every sample, "
					<< "batch_size was set to 1";
			}
			this->SetL2Norm(FLAGS_l2_norm);
			this->SetSqrt(FLAGS_sqrt);
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
				LOG_IF(INFO, (n  % 100 == 0)) << "Loaded " << n << " batches";
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
			if (sqrt_){
				caffe_powx<Dtype>(blob->count(), blob->mutable_cpu_data(),
					Dtype(0.5), blob->mutable_cpu_data());
			}
			if (l2_norm_){
				const int img_count = blob->count(1);
				Dtype* blob_data = blob->mutable_cpu_data();
				for (int n = 0; n < blob->num(); ++n){
					this->L2Normalize(img_count, blob_data);
					blob_data += img_count;
				}
			}
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
			Dtype* blob_data = blob->mutable_cpu_data();
			if (sqrt_){
				caffe_powx<Dtype>(blob->count(), blob_data, Dtype(0.5), blob_data);
			}
			const int img_count = blob->count(1);
			for (size_t i = 0; i < blob->shape(0); ++i){
				if (l2_norm_){
					this->L2Normalize(img_count, blob_data);
				}
				std::ofstream out_feat;
				string feat_path = paths[i] + ".feat";
				out_feat.open(feat_path.c_str(), std::ios::binary);
				CHECK(out_feat.is_open());
				out_feat.write((char*)(blob_data), sizeof(Dtype)* img_count);
				out_feat.close();
				blob_data += img_count;
			}
		}

	protected:
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
		bool sqrt_ = false;
		bool l2_norm_ = false;
	};
}
#endif