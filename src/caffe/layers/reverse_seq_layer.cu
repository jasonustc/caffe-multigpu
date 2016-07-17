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
		int outer_dim_ = bottom[0]->count(0, 2);
		int inner_dim_ = bottom[0]->count(2);
		for (int i = 0; i < outer_dim_; ++i){
			bottom_data += i * inner_dim_;
			top_data += index_data[i] * inner_dim_;
			caffe_copy(inner_dim_, bottom_data, top_data);
		}
	}

	template <typename Dtype>
	void ReverseSeqLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<bool>& propagate_down,
		const vector<Blob<Dtype>*>& top){
		if (propagate_down[0]){
			const Dtype* top_diff = top[0]->gpu_diff();
			Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
			const int* index_data = index_.cpu_data();
			for (int i = 0; i < outer_dim_; ++i){
				bottom_diff += i * inner_dim_;
				top_diff += index_data[i] * inner_dim_;
				caffe_copy(inner_dim_, top_diff, bottom_diff);
			}
		}
	}

	INSTANTIATE_LAYER_GPU_FUNCS(ReverseSeqLayer);
} //namespace caffe