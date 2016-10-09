#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/layer_factory.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/latent_loss_layer.hpp"

namespace caffe{

	template <typename Dtype>
	void LatentLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top){
		const int count = bottom[0]->count();
		const Dtype* mu_data = bottom[0]->gpu_data();
		const Dtype* sigma_data = bottom[1]->gpu_data();
		Dtype* log_sqr_sigma_data = log_square_sigma_.mutable_gpu_data();
		Dtype mu_sqr_sum;
		caffe_gpu_dot(count, mu_data, mu_data, &mu_sqr_sum);
		Dtype sigma_sqr_sum;
		caffe_gpu_dot(count, sigma_data, sigma_data, &sigma_sqr_sum);
		// log(x^2) = 2 * log(x)
		caffe_gpu_sqr(count, sigma_data, log_sqr_sigma_data);
		caffe_gpu_log(count, log_sqr_sigma_data, log_sqr_sigma_data);
		Dtype log_sigma_sum;
		caffe_gpu_dot(count, log_sqr_sigma_data, sum_multiplier_.gpu_data(), &log_sigma_sum);
		Dtype loss = (mu_sqr_sum + sigma_sqr_sum - log_sigma_sum) / bottom[0]->num() / Dtype(2);
		loss -= bottom[0]->count() / bottom[0]->num() / Dtype(2);
		top[0]->mutable_cpu_data()[0] = loss;
	}

	template <typename Dtype>
	__global__ void LatentLoss_Backward_kernel(const int n, const Dtype coeff, const Dtype* sigma_data,
		Dtype* sigma_diff_data){
		CUDA_KERNEL_LOOP(index, n){
			//TODO: check if data is not zero
			//first load to local device memory to save some time
			Dtype sig = max(sigma_data[index], Dtype(FLT_MIN));
			sigma_diff_data[index] = coeff *(sig - 1 / sig);
		}
	}

	template <typename Dtype>
	void LatentLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down,
		const vector<Blob<Dtype>*>& bottom){
		const int count = bottom[0]->count();
		const Dtype* mu_data = bottom[0]->gpu_data();
		Dtype* mu_diff = bottom[0]->mutable_gpu_diff();
		const Dtype* sigma_data = bottom[1]->gpu_data();
		Dtype* sigma_diff = bottom[1]->mutable_gpu_diff();
		const Dtype alpha = top[0]->cpu_diff()[0] / bottom[0]->num();
		if (propagate_down[0]){
			caffe_gpu_axpby(count, alpha, mu_data, Dtype(0.), mu_diff);
		}
		if (propagate_down[1]){
			LatentLoss_Backward_kernel<Dtype> << <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS >> >
				(count, alpha, sigma_data, sigma_diff);
		}
	}

	INSTANTIATE_LAYER_GPU_FUNCS(LatentLossLayer);
}