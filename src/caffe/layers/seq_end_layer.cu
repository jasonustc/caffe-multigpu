#include <utility>
#include <vector>

#include "caffe/layers/seq_end_layer.hpp"

namespace caffe{
	template <typename Dtype>
	void SeqEndLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top){
		this->InferSeqEndId(bottom, top);
		const Dtype* bottom_data = bottom[0]->gpu_data();
		Dtype* top_data = top[0]->mutable_gpu_data();
		int num_seq = end_id_.size();
		int outer_dim = bottom[0]->count(1);
		const Dtype* bottom_offset;
		for (int i = 0; i < num_seq; ++i){
			bottom_offset = bottom_data + outer_dim * static_cast<int>(end_id_[i]);
			caffe_copy(outer_dim, bottom_offset, top_data);
			top_data += outer_dim;
		}
	}

	template <typename Dtype>
	void SeqEndLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom){
		if (propagate_down[0]){
			const Dtype* top_diff = top[0]->gpu_diff();
			Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
			int num_seq = end_id_.size();
			int outer_dim = bottom[0]->count(1);
			Dtype* bottom_offset;
			for (int i = 0; i < num_seq; ++i){
				bottom_offset = bottom_diff + outer_dim * static_cast<int>(end_id_[i]);
				caffe_copy(outer_dim, top_diff, bottom_offset);
				top_diff += outer_dim;
			}
		}
	}

	INSTANTIATE_LAYER_GPU_FUNCS(SeqEndLayer);
}//namespace caffe