/********************************************************************************
** Copyright(c) 2016 USTC Reserved.
** auth: Xu Shen
** mail: shenxu@mail.ustc.edu.cn
** date: 2016/1/1
** desc: SimMergeLayer(CPU), merge similar feature maps and re-initialize similar
**       weights to learn more independent feature maps
*********************************************************************************/
#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/sim_merge_layer.hpp"

namespace caffe{

template <typename Dtype>
void SimMergeLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top){
	CHECK(this->layer_param_.sim_merge_param().has_weight_shape())
		<< "specified shape of weight should be provided";
	//specified operations for similarity based feature map merge
	iter_ = this->layer_param_.sim_merge_param().iter();
	threshold_ = this->layer_param_.sim_merge_param().threshold();
	axis_ = this->layer_param_.sim_merge_param().axis();
	curr_iter_ = 0;
	bias_term_ = this->layer_param_.sim_merge_param().has_bias_shape();
	CHECK_EQ(1 + bias_term_, this->layer_param_.param_size())
		<< "Number of fillers must be equal to number of shared parameters";
	this->blobs_.resize(1 + bias_term_);
	const BlobShape weight_shape = this->layer_param_.sim_merge_param().weight_shape();
	//build param blobs with exact shape to help check
	//w.r.t its source paramters
	this->blobs_[0].reset(new Blob<Dtype>(weight_shape));
	CHECK_GE(weight_shape.dim_size(), axis_);
	//feature map size
	N_ = bottom[0]->shape(axis_ + 1);
	//number of feature maps for one sample
	M_ = bottom[0]->shape(axis_);
	//number of samples
	K_ = bottom[0]->count(0, axis_);
	if (bias_term_){
		const BlobShape bias_shape = this->layer_param_.sim_merge_param().bias_shape();
		this->blobs_[1].reset(new Blob<Dtype>(bias_shape));
		CHECK_EQ(N_, this->blobs_[1]->count()) << "Number of bias parameters should be" 
			<< " equal to number of outputs" ;
	}
}

template <typename Dtype>
void SimMergeLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top){
	//In this layer we don't need to generate new output, so just
	//share data and diffs with its bottom
	top[0]->ReshapeLike(*bottom[0]);
	top[0]->ShareData(*bottom[0]);
	top[0]->ShareDiff(*bottom[0]);
	//count of feature maps
	CHECK_GT(N_, 1) << "Only more than 1 features need to be merged";
	this->sim_.Reshape(1, 1, 1, M_ * M_);
	//to store the intermediate similarities
	vector<int> temp_shape( N_, M_ * M_);
	this->temp_sim_.Reshape(temp_shape);
	//set up sum multiplier
	vector<int> batch_shape(1, N_);
	sum_multiplier_.Reshape(batch_shape);
	caffe_set(N_, Dtype(1.), sum_multiplier_.mutable_cpu_data());
}

//TODO: maybe this operation will be very time consuming, we 
// need to figure out a more efficient way
template <typename Dtype>
void SimMergeLayer<Dtype>::update_sim_matrix_cpu(const vector<Blob<Dtype>*>& bottom){
	const Dtype* bottom_data = bottom[0]->cpu_data();
	//because bottom diff will not be used in forward pass
	//we can use it as buffer
	Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
	Dtype* temp_sim_data = this->temp_sim_.mutable_cpu_data();
	Dtype* temp_sim_diff = this->temp_sim_.mutable_cpu_diff();
	//to save memory, put history similarity in data
	//and current similarity in diff
	Dtype* curr_sim_data = this->sim_.mutable_cpu_diff();
	Dtype* his_sim_data = this->sim_.mutable_cpu_data();
	const int offset = bottom[0]->offset(1);
	for (int i = 0; i < N_; ++i){
		//compute current similarity between feature maps
		caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, M_, K_, Dtype(1.),
			bottom_data + i * offset, bottom_data + i * offset, Dtype(0.), 
			temp_sim_data + i * offset);
		caffe_scal<Dtype>(offset, sqrt<Dtype>(*(temp_sim_data + i * offset)),
			temp_sim_data + i * offset);
	}
	//average accross batch
	caffe_cpu_gemv<Dtype>(CblasTrans, N_ , M_ * M_, Dtype(1.) / Dtype(N_), 
		temp_sim_data, sum_multiplier_.mutable_cpu_data(), 
		Dtype(0.), curr_sim_data);
	//update history similarity with current similarity
	const Dtype curr_iter = 1 + this->curr_iter_;
	caffe_cpu_axpby(count, (Dtype)1. / (Dtype)curr_iter, curr_sim_data, 
		(Dtype)this->curr_iter_ / (Dtype)curr_iter, his_sim_data);
}

//merge feature map n to feature map m
template <typename Dtype>
void SimMergeLayer<Dtype>::merge_two_feature_maps_cpu(const vector<Blob<Dtype>*>& top,
	const int m, const int n, const Dtype sim){
	const int num = top[0]->count(0, this->axis_);
	const int offset = top[0]->count(this->axis_);
	const int feat_dim = top[0]->count(this->axis_ + 1);
	Dtype* top_data = top[0]->mutable_cpu_data();
	const Dtype denom = 1 + sim;
	for (int i = 0; i < num; i++){
		Dtype* map_m_data = top_data + i * offset + m * feat_dim;
		const Dtype* map_n_data = top_data + i * offset + n * feat_dim;
		caffe_cpu_axpby(feat_dim, Dtype(sim) / denom, map_n_data, 
			Dtype(1.) / denom, map_m_data);
	}
}

//Reset weights/bias data to random initialized and 
//reseet weight/bias diff to 0.
//Here we assume that weight is in [num_out, ...] format
//and bias is in [num_out, ...] format
template <typename Dtype>
void SimMergeLayer<Dtype>::refresh_weight_cpu(const int j){
	Dtype* weight_data = this->blobs_[0]->mutable_cpu_data();
	Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();
	const int num = this->blobs_[0]->num();
	const int dim = this->blobs_[0]->count() / num;
	Dtype* weight_offset_data = weight_data + this->blobs_[0]->offset(j);
	Dtype* weight_offset_diff = weight_diff + this->blobs_[0]->offset(j);
	this->weight_filler_->Fill(weight_offset_data, dim);
	caffe_set(dim, (Dtype)0., weight_offset_diff);
	if (bias_term_){
		const int bias_dim = this->blobs_[1]->count(1);
		Dtype* bias_offset_data = this->blobs_[1]->mutable_cpu_data() + 
			this->blobs_[1]->offset(j);
		Dtype* bias_offset_diff = this->blobs_[1]->mutable_cpu_diff() + 
			this->blobs_[1]->offset(j);
		this->bias_filler_->Fill(bias_offset_data, bias_dim);
		caffe_set<Dtype>(bias_dim, (Dtype)0., bias_offset_diff);
	}
}

template <typename Dtype>
void SimMergeLayer<Dtype>::merge_sim_feature_maps_cpu(const vector<Blob<Dtype>*>& top){
	const int channel = top[0]->shape(this->axis_);
	const Dtype* sim_data = this->sim_.cpu_data();
	int index = 0;
	vector<int> merged_map_ids;
	for (int i = 0; i < channel; i++){
		//if map i has already been merged, we just skip it
		if (std::find(merged_map_ids.begin(), merged_map_ids.end(), i)
			!= merged_map_ids.end()){
			continue;
		}
		for (int j = i + 1; j < channel; j++){
			if (sim_data[index] > this->threshold_){
				merged_map_ids.push_back(j);
				this->merge_two_feature_maps_cpu(top, i, j, sim_data[index]);
				if (weight_term_){
					//re-initialize the weight
					this->refresh_weight_cpu(j);
				}
			}
			index++;
		}
	}
}

template <typename Dtype>
void SimMergeLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
	if (this->phase_ == TRAIN){
		this->update_sim_matrix_cpu(bottom);
		this->curr_iter_++;
		if (this->curr_iter_ % this->iter_ == 0){
			//reset number of iterations, 
			//so as to reset similarity matrix to all 0s
			this->curr_iter_ = 0;
			this->merge_sim_feature_maps_cpu(bottom);
		}
	}
}

template <typename Dtype>
void SimMergeLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	//currently, we have nothing to do
}

#ifdef CPU_ONLY
STUB_GPU(SimMergeLayer);
#endif

INSTANTIATE_CLASS(SimMergeLayer);
REGISTER_LAYER_CLASS(SimMerge);
}