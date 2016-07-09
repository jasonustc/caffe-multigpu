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
		const int* forward_index = index_.cpu_data();
		Dtype* top_offset;
		int inner_dim = bottom[0]->count(0, 2);
		int outer_dim = bottom[0]->count(2);
		for (int i = 0; i < inner_dim; ++i){
			bottom_data += i * outer_dim;
			top_offset = top_data + static_cast<int>(forward_index[i]) * outer_dim;
			caffe_copy(outer_dim, bottom_data, top_offset);
		}
	}

	template <typename Dtype>
	void ReverseSeqLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<bool>& propagate_down,
		const vector<Blob<Dtype>*>& top){
		if (propagate_down[0]){
			const int inner_dim = bottom[0]->count(0, 2);
			const int outer_dim = bottom[0]->count(2);
			const Dtype* top_diff = top[0]->gpu_diff();
			Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
			const int* backward_index = index_.cpu_diff();
			Dtype* bottom_offset;
			for (int i = 0; i < inner_dim; ++i){
				top_diff += i * outer_dim;
				bottom_offset = bottom_diff + static_cast<int>(backward_index[i]) 
					* outer_dim;
				caffe_copy(outer_dim, top_diff, bottom_offset);
			}
		}
	}

	INSTANTIATE_LAYER_GPU_FUNCS(ReverseSeqLayer);
} //namespace caffe