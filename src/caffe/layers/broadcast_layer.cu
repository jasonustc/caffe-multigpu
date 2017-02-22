#include <vector>

#include "caffe/layers/broadcast_layer.hpp"

namespace caffe{
	
	template <typename Dtype>
	void BroadcastLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top){
		int Ssize = bottom[0]->shape(axis_);
		int Tsize = bottom[1]->shape(axis_);
		int n = Tsize / Ssize;
		int num = bottom[0]->count(0, axis_);
		int count = bottom[0]->count(axis_);
		const Dtype* source_data = bottom[0]->gpu_data();
		Dtype* target_data = top[0]->mutable_gpu_data();
		for (int k = 0; k < num; ++k){
			for (int i = 0; i < n; ++i){
				caffe_copy(count, source_data, target_data);
				target_data += count;
			}
			source_data += count;
		}
	}

	template <typename Dtype>
	void BroadcastLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down,
		const vector<Blob<Dtype>*>& bottom){
		if (propagate_down[0]){
			int Ssize = bottom[0]->shape(axis_);
			int Tsize = bottom[1]->shape(axis_);
			int n = Tsize / Ssize;
			int num = bottom[0]->count(0, axis_);
			int count = bottom[0]->count(axis_);
			const Dtype* top_diff = top[0]->gpu_diff();
			Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
			for (int k = 0; k < num; ++k){
				// reset first
				caffe_copy(count, top_diff, bottom_diff);
				top_diff += count;
				for (int i = 1; i < n; ++i){
					caffe_gpu_axpy(count, Dtype(1.), top_diff, bottom_diff);
					top_diff += count;
				}
				bottom_diff += count;
			}
		}
	}

INSTANTIATE_LAYER_GPU_FUNCS(BroadcastLayer);
} // namespace caffe
