#ifndef CAFFE_SIM_MERGE_LAYER_HPP_
#define CAFFE_SIM_MERGE_LAYRE_HPP_

#include<vector>
#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe{

/**
 * Input: Blob[N, C, H, W]
 * We must set name to params so that we can reset them
 * Output: Blob[N, C, H, W]
 **/
template <typename Dtype>
class SimMergeLayer : public Layer<Dtype>{
public: 
	explicit SimMergeLayer(const LayerParameter& param)
		: Layer<Dtype>(param){}
	virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);
	virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);
	virtual inline const char* type() const { return "SimMerge"; }
	virtual inline int ExactNumBottomBlobs() const { return 1; }
	virtual inline int ExactNumTopBlobs() const { return 1; }

private:
	virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);
	virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);
	virtual void Backward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<bool>& propagate_down,
		const vector<Blob<Dtype>*>& top);
	virtual void Backward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<bool>& propagate_down,
		const vector<Blob<Dtype>*>& top);

	void update_sim_matrix_cpu();

	/**
	 * if the similarity of w_i and w_j exceeds threshold_
	 * merge weight_i and w_j based on the similarity
	 * and re-initialize w_j randomly
	 **/
	void merge_sim_weights_cpu();

	/**
	 * re-initialize w_j randomly
	 **/
	void refresh_weight_cpu(const int j);

	/**
	 * The similarity between w_i and w_j is computed by:
	 * sim_ij = w_i \dot w_j / (||w_i|| * ||w_j||)
	 **/
	void update_sim_matrix_gpu();

	//dim parameters
	//flatten the weights into a N_ x K_ matrix
	//N_: number of output K_: dim of single weight
	int N_;
	int K_;
	// the number of iterations to be monitored
	int iter_;
	// the number of similarities that are accumulated
	int curr_iter_;
	// the threshold of similarity to merge two feature map
	Dtype threshold_;

	//along which axis to merge
	int axis_;

	bool bias_term_;
	bool use_history_;
	// n x n  matrix to store similarities
	Blob<Dtype> sim_;
	//weight filler
	shared_ptr<Filler<Dtype>> weight_filler_;
	//bias filler
	shared_ptr<Filler<Dtype>> bias_filler_;
};

}

#endif