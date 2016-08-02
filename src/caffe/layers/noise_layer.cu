#include <vector>
#include <utility>

#include "caffe/layers/noise_layer.hpp"

namespace caffe{
	template <typename Dtype>
	void NoiseLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top){
		const Dtype* bottom_data = bottom[0]->gpu_data();
		Dtype* top_data = top[0]->mutable_gpu_data();
		Dtype* noise_data = noise_.mutable_gpu_data();
		const int count = bottom[0]->count();
		if (this->phase_ == TRAIN){
			//random noise
			if (noise_type_ == NoiseParameter_NoiseType_UNIFORM){
				caffe_gpu_rng_uniform(count, alpha_, beta_, noise_data);
			}
			else if (noise_type_ == NoiseParameter_NoiseType_GAUSSIAN){
				caffe_gpu_rng_gaussian(count, alpha_, beta_, noise_data);
			}
			else{
				LOG(FATAL) << "unknown noise type";
			}
			if (apply_type_ == NoiseParameter_ApplyType_MULTIPLY){
				caffe_gpu_mul(count, bottom_data, noise_data, top_data);
			}
			else if (apply_type_ == NoiseParameter_ApplyType_ADD){
				caffe_gpu_add(count, bottom_data, noise_data, top_data);
			}
			else{
				LOG(FATAL) << "unkown apply type";
			}
		}
		else{
			caffe_copy(count, bottom_data, top_data);
		}
	}

	template <typename Dtype>
	void NoiseLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<bool>& propagate_down,
		const vector<Blob<Dtype>*>& top){
		Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
		const Dtype* top_diff = top[0]->gpu_diff();
		const Dtype* noise_data = noise_.gpu_data();
		const int count = bottom[0]->count();
		if (apply_type_ == NoiseParameter_ApplyType_MULTIPLY){
			caffe_mul(count, top_diff, noise_data, bottom_diff);
		}
		else{
			caffe_copy(count, top_diff, bottom_diff);
		}
	}

	INSTANTIATE_LAYER_GPU_FUNCS(NoiseLayer);
} // namespace caffe