#include <vector>

#include "caffe/layers/switch_layer.hpp"

namespace caffe{
	template <typename Dtype>
	void SwitchLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top){
		if (has_bottom_index_){ // got input index
			bottom_index_ = bottom[bottom.size() - 1]->cpu_data()[0];
		}
		else{
			bottom_index_ = Rand(bottom.size());
			// for gtest 
//			bottom_index_ = 2;
		}
		caffe_copy(top[0]->count(), bottom[bottom_index_]->gpu_data(),
			top[0]->mutable_gpu_data());
		if (top.size() > 1){
			// output bottom index
			top[1]->mutable_cpu_data()[0] = bottom_index_;
		}
	}

	template <typename Dtype>
	void SwitchLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down,
		const vector<Blob<Dtype>*>& bottom){
		if (propagate_down[bottom_index_]){
			caffe_copy(top[0]->count(),
				top[0]->gpu_diff(),
				bottom[bottom_index_]->mutable_gpu_diff());
		}
	}

	INSTANTIATE_LAYER_GPU_FUNCS(SwitchLayer);
} // namespace caffe