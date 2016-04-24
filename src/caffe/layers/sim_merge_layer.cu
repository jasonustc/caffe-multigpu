
/********************************************************************************
** Copyright(c) 2016 USTC Reserved.
** auth: Xu Shen
** mail: shenxu@mail.ustc.edu.cn
** date: 2016/1/1
** desc: SimMergeLayer(GPU), merge similar feature maps and re-initialize similar
**       weights to learn more independent feature maps
*********************************************************************************/
#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/sim_merge_layer.hpp"

namespace caffe{

	template <typename Dtype>
	__global__ void ComputeSim(const int count, const int N, Dtype *sim_data){
		CUDA_KERNEL_LOOP(index, count){
			const int row = index / N;
			const int col = index % N;
			//sim(\vec{a}, \vec{b}) = (\vec{a} \dot \vec{b}) / 
			//(\sqrt(\vec{a} \dot \vec{a}) \times \sqrt(\vec{b} \dot \vec{b})
			const Dtype sqrt_i = sqrt(sim_data[row * N + row]);
			const Dtype sqrt_j = sqrt(sim_data[col * N + col]);
			sim_data[row * N + col] /= (sqrt_i * sqrt_j);
		}
	}

	//TODO: maybe this operation will be very time consuming, we 
	// need to figure out a more efficient way
	template <typename Dtype>
	void SimMergeLayer<Dtype>::update_sim_matrix_gpu(){
		Dtype* weight_data = this->blobs_[0]->mutable_gpu_data();
		//to save memory, put history similarity in data
		//and current similarity in diff
		Dtype* curr_sim_data = this->sim_.mutable_gpu_diff();
		Dtype* his_sim_data = this->sim_.mutable_gpu_data();
		caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, N_, N_, K_, Dtype(1.),
			weight_data, weight_data, Dtype(0), curr_sim_data);
		const int count = N_ * N_;
		ComputeSim<Dtype><<< CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
			count, N_, curr_sim_data);
		CUDA_POST_KERNEL_CHECK;
		//update history similarity with current similarity
		if (use_history_){
			const Dtype curr_iter = 1 + this->curr_iter_;
			caffe_gpu_axpby(N_ * N_, (Dtype)1. / (Dtype)curr_iter, curr_sim_data,
				(Dtype)this->curr_iter_ / (Dtype)curr_iter, his_sim_data);
		}
		else{
			caffe_copy<Dtype>(N_ * N_, curr_sim_data, his_sim_data);
		}
	}

	template <typename Dtype>
	void SimMergeLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		//currently, we have nothing to do
	}

	template <typename Dtype>
	void SimMergeLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
		this->update_sim_matrix_gpu();
		this->curr_iter_++;
		if (this->curr_iter_ % this->iter_ == 0){
			//reset number of iterations, 
			//so as to reset similarity matrix to all 0s
			this->curr_iter_ = 0;
			// NOTE: I don't think a gpu version can accelerate the computation
			// so I just use the cpu code here
			this->merge_sim_weights_cpu();
		}
	}

	INSTANTIATE_LAYER_GPU_FUNCS(SimMergeLayer);
}
