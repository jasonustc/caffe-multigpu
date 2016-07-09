#include <vector>
#include <utility>

#include "caffe/layers/reverse_seq_layer.hpp"

namespace caffe{

	template <typename Dtype>
	void ReverseSeqLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top){
		CHECK_EQ(bottom[1]->num_axes(), 2);
		CHECK_EQ(bottom[0]->shape(0), bottom[1]->shape(0));
		CHECK_EQ(bottom[0]->shape(1), bottom[1]->shape(1));
	}

	template <typename Dtype>
	void ReverseSeqLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top){
		// T_, #streams
		CHECK_EQ(bottom[1]->num_axes(), 2);
		CHECK_EQ(bottom[0]->shape(0), bottom[1]->shape(0));
		CHECK_EQ(bottom[0]->shape(1), bottom[1]->shape(1));
		top[0]->ReshapeLike(*(bottom[0]));
	}

	template <typename Dtype>
	void ReverseSeqLayer<Dtype>::GetIndex(const vector<Blob<Dtype>*>& bottom){
		const Dtype* cont_data = bottom[1]->cpu_data();
		int* forward_index = index_.mutable_cpu_data();
		int* backward_index = index_.mutable_cpu_diff();
		int seq_begin;
		const int T = bottom[1]->shape(0);
		const int N = bottom[1]->shape(1);
		for (int n = 0; n < N; ++n){
			seq_begin = 0;
			for (int t = 1; t < T; ++t){
				if (cont_data[t * T + n] == 0){
					int seq_len = t - seq_begin;
					for (int i = 0; i < seq_len; ++i){
						int top_t = seq_len - i - 1 + seq_begin;
						forward_index[t * T + n] = top_t * N + n;
						backward_index[top_t * N + n] = t * T + n;
					}
				}
			}
		}
	}

	template <typename Dtype>
	void ReverseSeqLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top){
		const Dtype* bottom_data = bottom[0]->cpu_data();
		Dtype* top_data = top[0]->mutable_cpu_data();
		const int* forward_index = index_.cpu_data();
		this->GetIndex(bottom);
		const int inner_dim = bottom[0]->count(0, 2);
		const int outer_dim = bottom[0]->count(2);
		Dtype* top_offset;
		for (int i = 0; i < inner_dim; ++i){
			bottom_data += i * outer_dim;
			top_offset = top_data + static_cast<int>(forward_index[i]) * outer_dim;
			caffe_copy(outer_dim, bottom_data, top_offset);
		}
	}

	template <typename Dtype>
	void ReverseSeqLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom){
		if (propagate_down[0]){
			const int inner_dim = bottom[0]->count(0, 2);
			const int outer_dim = bottom[0]->count(2);
			const Dtype* top_diff = top[0]->cpu_diff();
			Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
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

#ifdef CPU_ONLY
	STUB_GPU(ReverseSeqLayer);
#endif

	INSTANTIATE_CLASS(ReverseSeqLayer);
	REGISTER_LAYER_CLASS(ReverseSeq);
}