#include <vector>
#include <utility>

#include "caffe/layers/sampling_layer.hpp"

namespace caffe{

	template <typename Dtype>
	void SamplingLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top){
		const Dtype* mu_data = bottom[0]->gpu_data();
		const Dtype* sigma_data = bottom[1]->gpu_data();
		Dtype* top_data = top[0]->mutable_gpu_data();
		const int count = bottom[0]->count();
		Dtype* gaussian_data = gaussian_value_.mutable_gpu_data();
		caffe_gpu_rng_gaussian(count, Dtype(0.), Dtype(1.), gaussian_data);
		caffe_gpu_mul(count, sigma_data, gaussian_data, top_data);
		caffe_gpu_add(count, mu_data, top_data, top_data);
	}

	template <typename Dtype>
	void SamplingLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down,
		const vector<Blob<Dtype>*>& bottom){
		const int count = bottom[0]->count();
		Dtype* mu_diff = bottom[0]->mutable_gpu_diff();
		Dtype* sigma_diff = bottom[1]->mutable_gpu_diff();
		const Dtype* top_diff = top[0]->gpu_diff();
		const Dtype* gaussian_data = gaussian_value_.gpu_data();
		if (propagate_down[0]){
			caffe_copy(count, top_diff, mu_diff);
		}
		if (propagate_down[1]){
			caffe_gpu_mul(count, top_diff, gaussian_data, sigma_diff);
		}
	}

	INSTANTIATE_LAYER_GPU_FUNCS(SamplingLayer);
} //namespace caffe
