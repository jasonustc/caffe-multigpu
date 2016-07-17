#include <vector>
#include <utility>

#include "caffe/layers/reverse_seq_layer.hpp"

namespace caffe{

	template <typename Dtype>
	void ReverseSeqLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top){
		CHECK_EQ(bottom[1]->num_axes(), 2);
		CHECK_EQ(bottom[0]->shape(0), bottom[1]->shape(0));
		for (int i = 0; i < bottom.size(); ++i){
			CHECK_EQ(bottom[i]->shape(1), 1)
				<< "Currently, we only support 1 stream data processing";
		}
	}

	template <typename Dtype>
	void ReverseSeqLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top){
		// T_
		CHECK_EQ(bottom[1]->num_axes(), 2);
		CHECK_EQ(bottom[0]->shape(0), bottom[1]->shape(0));
		for (int i = 0; i < bottom.size(); ++i){
			CHECK_EQ(bottom[i]->shape(1), 1)
				<< "Currently, we only support 1 stream data processing";
		}
		top[0]->ReshapeLike(*(bottom[0]));
		vector<int> index_shape = bottom[1]->shape();
		index_.Reshape(index_shape);
	}

	template <typename Dtype>
	void ReverseSeqLayer<Dtype>::GetIndex(const vector<Blob<Dtype>*>& bottom){
		const Dtype* cont_data = bottom[1]->cpu_data();
		int* index_data = index_.mutable_cpu_data();
		int seq_begin = 0;
		const int T = bottom[1]->shape(0);
		for (int t = 1; t < T; ++t){
			if (cont_data[t] == 0){
				int seq_len = t - seq_begin;
				for (int i = 0; i < seq_len; ++i){
					int top_t = seq_len - i - 1 + seq_begin;
					index_data[seq_begin + i] = top_t;
				}
				seq_begin = t;
			}
		}
		// last sequence
		if (seq_begin < (T - 1)){
			int seq_len = T - seq_begin;
			for (int i = 0; i < seq_len; ++i){
				int top_t = seq_len - i - 1 + seq_begin;
				index_data[seq_begin + i] = top_t;
			}
		}
	}

	template <typename Dtype>
	void ReverseSeqLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top){
		const Dtype* bottom_data = bottom[0]->cpu_data();
		Dtype* top_data = top[0]->mutable_cpu_data();
		this->GetIndex(bottom);
		const int outer_dim = bottom[0]->count(0, 2);
		const int inner_dim = bottom[0]->count(2);
		const int* index_data = index_.cpu_data();
		Dtype* top_offset;
		for (int i = 0; i < outer_dim; ++i){
			top_offset = top_data + index_data[i] * inner_dim;
			caffe_copy<Dtype>(inner_dim, bottom_data, top_offset);
			bottom_data += inner_dim;
		}
	}

	template <typename Dtype>
	void ReverseSeqLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom){
		if (propagate_down[0]){
			const Dtype* top_diff = top[0]->cpu_diff();
			Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
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

#ifdef CPU_ONLY
	STUB_GPU(ReverseSeqLayer);
#endif

	INSTANTIATE_CLASS(ReverseSeqLayer);
	REGISTER_LAYER_CLASS(ReverseSeq);
}