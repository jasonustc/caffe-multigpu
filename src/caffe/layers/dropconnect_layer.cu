#include <vector>

#include "caffe/layers/dropconnect_layer.hpp"

namespace caffe{
	template <typename Dtype>
	__global__ void DropWeight(const int n, const Dtype* in,
		const unsigned int* mask, const unsigned int threshold, const float scale,
		Dtype* out){
		//what's the usage of index here?
		CUDA_KERNEL_LOOP(index, n){
			out[index] = in[index] * (mask[index] > threshold) * scale;
		}
	}

	template <typename Dtype>
	void DropConnectLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		const Dtype* bottom_data = bottom[0]->gpu_data();
		Dtype* top_data = top[0]->mutable_gpu_data();
		const Dtype* weight = this->blobs_[0]->gpu_data();
		Dtype* dropped_weight = this->dropped_weight_.mutable_gpu_data();
		const int count = this->blobs_[0]->count();
		if (this->phase_ == TRAIN){
			unsigned int* weight_multiplier =
				static_cast<unsigned int*>(weight_multiplier_.mutable_gpu_data());
			caffe_gpu_rng_uniform(count, weight_multiplier);
			DropWeight<Dtype> << <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS >> >(
				count, weight, weight_multiplier, uint_thres_, scale_, dropped_weight);
		}
		else{
			caffe_copy(count, weight, dropped_weight);
		}
		caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, N_, K_, (Dtype)1.,
			bottom_data, dropped_weight, (Dtype)0., top_data);
		if (bias_term_) {
			caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, 1, (Dtype)1.,
				bias_multiplier_.gpu_data(),
				this->blobs_[1]->gpu_data(), (Dtype)1., top_data);
		}
	}

	template <typename Dtype>
	void DropConnectLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down,
		const vector<Blob<Dtype>*>& bottom) {
		if (this->param_propagate_down_[0]) {
			const Dtype* top_diff = top[0]->gpu_diff();
			const Dtype* bottom_data = bottom[0]->gpu_data();
			Dtype* weight_diff = this->blobs_[0]->mutable_gpu_diff();
			const int count = this->blobs_[0]->count();
			// Gradient with respect to weight
			caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, N_, K_, M_, (Dtype)1.,
				top_diff, bottom_data, (Dtype)0., this->blobs_[0]->mutable_gpu_diff());
			if (this->phase_ == TRAIN){
				unsigned int* weight_multiplier = static_cast<unsigned int*>(
					weight_multiplier_.mutable_gpu_data());
				DropWeight<Dtype> << <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS >> >(
					count, weight_diff, weight_multiplier, uint_thres_, scale_, weight_diff);
			}
		}
		if (bias_term_ && this->param_propagate_down_[1]) {
			const Dtype* top_diff = top[0]->gpu_diff();
			// Gradient with respect to bias
			caffe_gpu_gemv<Dtype>(CblasTrans, M_, N_, (Dtype)1., top_diff,
				bias_multiplier_.gpu_data(), (Dtype)0.,
				this->blobs_[1]->mutable_gpu_diff());
		}
		if (propagate_down[0]) {
			const Dtype* top_diff = top[0]->gpu_diff();
			// Gradient with respect to bottom data
			caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, K_, N_, (Dtype)1.,
				top_diff, this->dropped_weight_.gpu_data(), (Dtype)0.,
				bottom[0]->mutable_gpu_diff());
		}
	}

	INSTANTIATE_LAYER_GPU_FUNCS(DropConnectLayer);
}
