#include <vector>
#include <utility>

#include "caffe/layers/reverse_seq_layer.hpp"

namespace caffe{

	template <typename Dtype>
	void ReverseSeqLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top){
		const Dtype* bottom_data = bottom[0]->gpu_data();
		Dtype* top_data = top[0]->mutable_gpu_data();
		this->GetIndex(bottom);
		const int* index_data = index_.cpu_data();
		const int outer_dim = bottom[0]->count(0, 2);
		const int inner_dim = bottom[0]->count(2);
		Dtype* top_offset;
		for (int i = 0; i < outer_dim; ++i){
			top_offset = top_data + index_data[i] * inner_dim;
			caffe_copy(inner_dim, bottom_data, top_offset);
			bottom_data += inner_dim;
		}
	}

	template <typename Dtype>
	void ReverseSeqLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down,
		const vector<Blob<Dtype>*>& bottom){
		if (propagate_down[0]){
			const Dtype* top_diff = top[0]->gpu_diff();
			Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
			const int* index_data = index_.cpu_data();
			const int outer_dim = bottom[0]->count(0, 2);
			const int inner_dim = bottom[0]->count(2);
			const Dtype* top_diff_offset;
			for (int i = 0; i < outer_dim; ++i){
				top_diff_offset = top_diff + index_data[i] * inner_dim;
				caffe_copy(inner_dim, top_diff_offset, bottom_diff);
				bottom_diff += inner_dim;
			}
		}
	}

	INSTANTIATE_LAYER_GPU_FUNCS(ReverseSeqLayer);
} //namespace caffe