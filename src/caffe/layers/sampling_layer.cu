#include <vector>
#include <utility>

#include "caffe/layers/sampling_layer.hpp"

namespace caffe{

	template <typename Dtype>
	void SamplingLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top){
		Dtype* top_data = top[0]->mutable_gpu_data();
		const int count = bottom[0]->count();
		switch (sample_type_){
		case SamplingParameter_SampleType_GAUSSIAN:
		{
		  const Dtype* mu_data = bottom[0]->gpu_data();
		  const Dtype* sigma_data = bottom[1]->gpu_data();
		  Dtype* gaussian_data = gaussian_value_.mutable_gpu_data();
		  caffe_gpu_rng_gaussian(count, Dtype(0.), Dtype(1.), gaussian_data);
		  caffe_gpu_mul(count, sigma_data, gaussian_data, top_data);
		  caffe_gpu_add(count, mu_data, top_data, top_data);
		  break;
		}
		case SamplingParameter_SampleType_UNIFORM:
		{
		 const Dtype* a_data = bottom[0]->gpu_data();
		 const Dtype* b_data = bottom[1]->gpu_data();
		 CHECK_LT(bottom[0]->cpu_data()[0], bottom[1]->cpu_data()[0]);
		 caffe_gpu_rng_uniform(count, a_data, b_data, top_data);
		 break;
		}
		case SamplingParameter_SampleType_BERNOULLI:
		{
		   const Dtype* p_data = bottom[0]->gpu_data();
		   CHECK_GE(bottom[0]->cpu_data()[0], 0);
		   CHECK_LE(bottom[0]->cpu_data()[0], 1);
		   caffe_gpu_rng_bernoulli(count, p_data, top_data);
		   break;
		}
		}
	}

	template <typename Dtype>
	void SamplingLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down,
		const vector<Blob<Dtype>*>& bottom){
		if (is_gaussian_){
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
		else{
			LOG(FATAL) << "backward is not implemented for bernoulli and uniform sampling";
		}
	}

	INSTANTIATE_LAYER_GPU_FUNCS(SamplingLayer);
} //namespace caffe
