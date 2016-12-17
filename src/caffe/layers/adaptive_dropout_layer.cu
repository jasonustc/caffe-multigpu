#include <vector>
#include <string>

#include "caffe/layers/adaptive_dropout_layer.hpp"

namespace caffe{
	template <typename Dtype>
	__global__ void SigmoidActivate_gpu(const int n, const Dtype* in, Dtype* out){
		CUDA_KERNEL_LOOP(index, n){
			out[index] = 1. / (1. + exp(-in[index]));
		}
	}

	template <typename Dtype>
	__global__ void ReluActivate_gpu(const int n, const Dtype* in, Dtype* out){
		CUDA_KERNEL_LOOP(index, n){
			out[index] = in[index] > 0 ? in[index] : 0;
		}
	}

	template <typename Dtype>
	inline void activate_gpu(const int n, const Dtype* in, Dtype* out,
		AdaptiveDropoutParameter_ActType act_type){
		switch (act_type){
		case caffe::AdaptiveDropoutParameter_ActType_SIGMOID:
			SigmoidActivate_gpu<Dtype> << <CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS >> >(n, in, out);
			break;
		case caffe::AdaptiveDropoutParameter_ActType_RELU:
			ReluActivate_gpu<Dtype> << <CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS >> >(n, in, out);
			break;
		default:
			LOG(FATAL) << "Unkown activate function.";
		}
	}

	template <typename Dtype>
	__global__ void ad_axpb(const int n, const Dtype* in, Dtype* out,
		const Dtype alpha, const Dtype beta){
		CUDA_KERNEL_LOOP(index, n){
			out[index] = alpha * in[index] + beta;
		}
	}

	template <typename Dtype>
	void AdaptiveDropoutLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		const Dtype* bottom_data = bottom[0]->gpu_data();
		Dtype* top_data = top[0]->mutable_gpu_data();
		const Dtype* weight_data = this->blobs_[0]->gpu_data();
		Dtype* prob_data = this->prob_vec_.mutable_gpu_data();
		unsigned int *rand_vec_data = this->rand_vec_.mutable_gpu_data();
		const int count_weight = this->blobs_[0]->count();
		const int count_prob = this->prob_vec_.count();
		//prob_data = alpha * op(bottom_data) * (weight_data) + beta * prob_data
		caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, N_, K_, (Dtype)1.,
			bottom_data, weight_data, (Dtype)0., prob_data);
		if (bias_term_) {
			caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, 1, (Dtype)1.,
				bias_multiplier_.gpu_data(),
				this->blobs_[1]->gpu_data(), (Dtype)1., this->prob_vec_.mutable_gpu_data());
		}
		//prob_act = f(alpha*(pi * bottom + bias) + beta)
		ad_axpb<Dtype> << <CAFFE_GET_BLOCKS(count_prob), CAFFE_CUDA_NUM_THREADS >> >
			(count_prob, prob_vec_.gpu_data(), prob_data, alpha_, beta_);
		//activate probability
		activate_gpu<Dtype>(count_prob, prob_vec_.gpu_data(), prob_data, this->prob_act_type_);
		//compute hidden units
		caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, N_, K_, (Dtype)1.,
			bottom_data, weight_data, (Dtype)0., unact_hidden_.mutable_gpu_data());
		if (bias_term_) {
			caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, 1, (Dtype)1.,
				bias_multiplier_.gpu_data(),
				this->blobs_[1]->gpu_data(), (Dtype)1., unact_hidden_.mutable_gpu_data());
		}
		activate_gpu(top[0]->count(), unact_hidden_.gpu_data(), top_data, this->hidden_act_type_);
		CUDA_POST_KERNEL_CHECK;
		if (this->phase_ == TRAIN){
			//p[i] is the probability of r[i]=1
			caffe_gpu_rng_bernoulli<Dtype>(count_prob, prob_vec_.gpu_data(), rand_vec_.mutable_gpu_data());
			caffe_gpu_mul_b<Dtype>(count_prob, top[0]->gpu_data(), rand_vec_.gpu_data(),
				top[0]->mutable_gpu_data());
		}
		else{
			caffe_gpu_mul<Dtype>(count_prob, top[0]->gpu_data(), prob_data, top[0]->mutable_gpu_data());
		}
	}

	template<typename Dtype>
	__global__ void SigmoidBackward_gpu(const int n, const Dtype* in_diff,
		const Dtype* unact_data, Dtype* out_diff){
		CUDA_KERNEL_LOOP(index, n){
			const Dtype sigmoid_x = 1. / (1. + exp(-unact_data[index]));
			out_diff[index] = in_diff[index] * sigmoid_x * (1 - sigmoid_x);
		}
	}

	template <typename Dtype>
	__global__ void ReLUBackward_gpu(const int n, const Dtype* in_diff,
		const Dtype* in_data, Dtype* out_diff){
		CUDA_KERNEL_LOOP(index, n){
			out_diff[index] = in_diff[index] * (in_data[index] > 0);
		}
	}

	template <typename Dtype>
	inline void ActBackward_gpu(const int n, const Dtype* in_diff,
		const Dtype* in_data, Dtype* out_diff, AdaptiveDropoutParameter_ActType act_type){
		switch (act_type)
		{
		case caffe::AdaptiveDropoutParameter_ActType_RELU:
			ReLUBackward_gpu<Dtype > << <CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS >> >(
				n, in_diff, in_data, out_diff);
			break;
		case caffe::AdaptiveDropoutParameter_ActType_SIGMOID:
			SigmoidBackward_gpu<Dtype> << <CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS >> >(
				n, in_diff, in_data, out_diff);
			break;
		default:
			LOG(FATAL) << "unknown act function type.";
			break;
		}
	}

	template <typename Dtype>
	__global__ void DropoutBackward_gpu(const int n, const Dtype* in_diff,
		const unsigned int* mask, const float scale, Dtype* out_diff){
		CUDA_KERNEL_LOOP(index, n){
			out_diff[index] = in_diff[index] * scale * mask[index];
		}
	}

	/* Two choices in dropout:
	* 1. scale output by expection of dropout
	* 2. scale gradient by invert of expection in training
	*/

	template <typename Dtype>
	void AdaptiveDropoutLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down,
		const vector<Blob<Dtype>*>& bottom) {
		const int count_top = top[0]->count();
		Dtype* top_diff = top[0]->mutable_gpu_diff();
		Dtype* unact_hidden_diff = this->unact_hidden_.mutable_gpu_diff();
		//backward through dropout
		const unsigned int* rand_vec_data = this->rand_vec_.gpu_data();
		//top_diff = top_diff * rand_vec_data
		DropoutBackward_gpu<Dtype> << < CAFFE_GET_BLOCKS(count_top), CAFFE_CUDA_NUM_THREADS >> >(
			count_top, top_diff, rand_vec_data, (Dtype)1., prob_vec_.mutable_gpu_diff());
		//backward through non-linear activation
		const Dtype* in_data = unact_hidden_.gpu_data();
		ActBackward_gpu(count_top, prob_vec_.gpu_diff(), in_data, unact_hidden_diff, hidden_act_type_);

		if (this->param_propagate_down_[0]) {
			const Dtype* bottom_data = bottom[0]->gpu_data();
			// Gradient with respect to weight
			caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, N_, K_, M_, (Dtype)1.,
				unact_hidden_diff, bottom_data, (Dtype)0., this->blobs_[0]->mutable_gpu_diff());
		}
		if (bias_term_ && this->param_propagate_down_[1]) {
			// Gradient with respect to bias
			caffe_gpu_gemv<Dtype>(CblasTrans, M_, N_, (Dtype)1., unact_hidden_diff,
				bias_multiplier_.gpu_data(), (Dtype)0.,
				this->blobs_[1]->mutable_gpu_diff());
		}
		if (propagate_down[0]) {
			// Gradient with respect to bottom data
			caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, K_, N_, (Dtype)1.,
				unact_hidden_diff, this->blobs_[0]->gpu_data(), (Dtype)0.,
				bottom[0]->mutable_gpu_diff());
		}
	}

	INSTANTIATE_LAYER_GPU_FUNCS(AdaptiveDropoutLayer);
}