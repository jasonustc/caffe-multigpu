#include <vector>
#include <utility>

#include "caffe/layers/noise_layer.hpp"

namespace caffe{
	template <typename Dtype>
	void NoiseLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top){
		alpha_ = this->layer_param_.noise_param().alpha();
		beta_ = this->layer_param_.noise_param().beta();
		noise_type_ = this->layer_param_.noise_param().noise_type();
		apply_type_ = this->layer_param_.noise_param().apply_type();
	}

	template <typename Dtype>
	void NoiseLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top){
		top[0]->ReshapeLike(*(bottom[0]));
		noise_.ReshapeLike(*(bottom[0]));
	}

	template <typename Dtype>
	void NoiseLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top){
		const Dtype* bottom_data = bottom[0]->cpu_data();
		Dtype* top_data = top[0]->mutable_cpu_data();
		Dtype* noise_data = noise_.mutable_cpu_data();
		const int count = bottom[0]->count();
		if (this->phase_ == TRAIN){
			//random noise
			if (noise_type_ == NoiseParameter_NoiseType_UNIFORM){
				caffe_rng_uniform(count, alpha_, beta_, noise_data);
			}
			else if (noise_type_ == NoiseParameter_NoiseType_GAUSSIAN){
				caffe_rng_gaussian(count, alpha_, beta_, noise_data);
			}
			else{
				LOG(FATAL) << "unkown noise type";
			}
			if (apply_type_ == NoiseParameter_ApplyType_MULTIPLY){
				//elementwise operation
				caffe_mul(count, bottom_data, noise_data, top_data);
			}
			else if (apply_type_ == NoiseParameter_ApplyType_ADD){
				caffe_add(count, bottom_data, noise_data, top_data);
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
	void NoiseLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down,
		const vector<Blob<Dtype>*>& bottom){
		const int count = bottom[0]->count();
		Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
		const Dtype* top_diff = top[0]->cpu_diff();
		const Dtype* noise_data = noise_.cpu_data();
		if (this->apply_type_ == NoiseParameter_ApplyType_MULTIPLY){
			caffe_mul(count, top_diff, noise_data, bottom_diff);
		}
		else{
			caffe_copy(count, top_diff, bottom_diff);
		}
	}

#ifdef CPU_ONLY
	STUB_GPU(NoiseLayer);
#endif

	INSTANTIATE_CLASS(NoiseLayer);
	REGISTER_LAYER_CLASS(Noise);
} // namespace caffe