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
	use_history_ = this->layer_param_.sim_merge_param().use_history();
	CHECK_EQ(1 + bias_term_, this->layer_param_.param_size())
		<< "Number of fillers must be equal to number of shared parameters";
	this->blobs_.resize(1 + bias_term_);
	vector<int> weight_dim;
	const BlobShape weight_shape = this->layer_param_.sim_merge_param().weight_shape();
	for (int i = 0; i < weight_shape.dim_size(); ++i){
		weight_dim.push_back(weight_shape.dim(i));
	}
	//build param blobs with exact shape to help check
	//w.r.t its source paramters
	this->blobs_[0].reset(new Blob<Dtype>(weight_dim));
	weight_filler_.reset(GetFiller<Dtype>(this->layer_param_.sim_merge_param().weight_filler()));
//	CHECK_GE(weight_shape.dim_size(), axis_);
	//number of output
	//dim_0 * dim_1 * ... * dim_{axis_-1} is the number of outputs
	N_ = this->blobs_[0]->count(0, axis_);
	//number of weight dim for input
	K_ = bottom[0]->count(axis_);
	if (bias_term_){
		const BlobShape bias_shape = this->layer_param_.sim_merge_param().bias_shape();
		vector<int> bias_dim;
		for (int i = 0; i < bias_shape.dim_size(); ++i){
			bias_dim.push_back(bias_shape.dim(i));
		}
		this->blobs_[1].reset(new Blob<Dtype>(bias_dim));
		CHECK_EQ(N_, this->blobs_[1]->count()) << "Number of bias parameters should be" 
			<< " equal to number of outputs" ;
		bias_filler_.reset(GetFiller<Dtype>(this->layer_param_.sim_merge_param().bias_filler()));
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
	//count of weights
	CHECK_GT(N_, 1) << "Only more than 1 features need to be merged";
	this->sim_.Reshape(1, 1, 1, N_ * N_);
}

//TODO: maybe this operation will be very time consuming, we 
// need to figure out a more efficient way
template <typename Dtype>
void SimMergeLayer<Dtype>::update_sim_matrix_cpu(){
	const Dtype* weight_data = this->blobs_[0]->mutable_cpu_data();
	//to save memory, put history similarity in data
	//and current similarity in diff
	Dtype* curr_sim_data = this->sim_.mutable_cpu_diff();
	Dtype* his_sim_data = this->sim_.mutable_cpu_data();
	caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, N_, N_, K_, Dtype(1),
		weight_data, weight_data, Dtype(0), curr_sim_data);
	//sim(\vec{a}, \vec{b}) = (\vec{a} \dot \vec{b}) / 
	//(\sqrt(\vec{a} \dot \vec{a}) \times \sqrt(\vec{b} \dot \vec{b})
	for (int i = 0; i < N_; ++i){
		for (int j = i + 1; j < N_; ++j){
			const Dtype sqrt_i = sqrt(curr_sim_data[i * N_ + i]);
			const Dtype sqrt_j = sqrt(curr_sim_data[j * N_ + j]);
			curr_sim_data[i * N_ + j] /= (sqrt_i * sqrt_j);
			curr_sim_data[j * N_ + i] /= (sqrt_i * sqrt_j);
		}
	}
	// update history similarity with current similarity
	// NOTE: we need to check whether incorporate history 
	// similarity is better or worse
	if (use_history_){
		const Dtype curr_iter = 1 + this->curr_iter_;
		caffe_cpu_axpby(N_ * N_, (Dtype)1. / (Dtype)curr_iter, curr_sim_data,
			(Dtype)this->curr_iter_ / (Dtype)curr_iter, his_sim_data);
	}
	else{
		caffe_copy<Dtype>(N_ * N_, curr_sim_data, his_sim_data);
	}
}


//Reset weights/bias data to random initialized and 
//reseet weight/bias diff to 0.
//Here we assume that weight is in [N_, K_] format
//and bias is in [N_] format
template <typename Dtype>
void SimMergeLayer<Dtype>::refresh_weight_cpu(const int j){
	Dtype* weight_data = this->blobs_[0]->mutable_cpu_data();
	Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();
	Dtype* weight_data_j = weight_data + j * K_;
	Dtype* weight_diff_j = weight_diff + j * K_;
	this->weight_filler_->Fill(K_, weight_data_j);
	caffe_set(K_, (Dtype)0., weight_diff_j);
	if (bias_term_){
		Dtype* bias_data_j = this->blobs_[1]->mutable_cpu_data() + 
			j;
		Dtype* bias_diff_j = this->blobs_[1]->mutable_cpu_diff() + 
			j;
		this->bias_filler_->Fill(1, bias_data_j);
		caffe_set<Dtype>(1, (Dtype)0., bias_diff_j);
	}
}

/**
 * NOTE: we should merge weights in the backward pass
 * because diffs will be rewritten in bp, so it is non-sense
 * to refresh diffs in the forward pass
 **/
template<typename Dtype>
void SimMergeLayer<Dtype>::merge_sim_weights_cpu(){
	Dtype* weight_data = this->blobs_[0]->mutable_cpu_data();
	Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();
	const Dtype* sim_data = this->sim_.cpu_data();
	vector<int> merged_index;
	for (int i = 0; i < N_; ++i){
		if (std::find(merged_index.begin(), merged_index.end(), i)
			!= merged_index.end()){
			continue;
		}
		for (int j = i + 1; j < N_; ++j){
			if (std::find(merged_index.begin(), merged_index.end(), j)
				!= merged_index.end()){
				continue;
			}
			const Dtype sim_ij = sim_data[i * N_ + j];
			if (sim_ij > threshold_){
				//weight_i := (1 - sim_ij) * weight_i + sim_ij * weight_j
				caffe_cpu_axpby<Dtype>(K_, Dtype(sim_ij), weight_data + j * K_,
					Dtype(1 - sim_ij), weight_data + i * N_);
				//weight_diff_i := (1 - sim_ij) * weight_diff_i 
				// + sim_ij * weight_diff_j
				caffe_cpu_axpby<Dtype>(K_, Dtype(sim_ij), weight_diff + j * K_,
					Dtype(1 - sim_ij), weight_diff + i * K_);
				//refresh weight and diffs
				refresh_weight_cpu(j);
				merged_index.push_back(i);
				merged_index.push_back(j);
			}
		}
	}
}


template <typename Dtype>
void SimMergeLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
	//currently, we have nothing to do
}

template <typename Dtype>
void SimMergeLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	this->update_sim_matrix_cpu();
	this->curr_iter_++;
	if (this->curr_iter_ % this->iter_ == 0){
		//reset number of iterations, 
		//so as to reset similarity matrix to all 0s
		this->curr_iter_ = 0;
		this->merge_sim_weights_cpu();
	}
}

#ifdef CPU_ONLY
STUB_GPU(SimMergeLayer);
#endif

INSTANTIATE_CLASS(SimMergeLayer);
REGISTER_LAYER_CLASS(SimMerge);
}