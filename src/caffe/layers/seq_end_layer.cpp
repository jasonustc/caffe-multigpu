#include <utility>
#include <vector>

#include "caffe/layers/seq_end_layer.hpp"

namespace caffe{
	template <typename Dtype>
	void SeqEndLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top){
		CHECK_EQ(bottom[1]->num_axes(), 2);
		CHECK_EQ(bottom[0]->num_axes(), 3);
		if (bottom[0]->shape(1) > 1){
			LOG(ERROR) << "Please make sure that each stream uses the same cont variable";
		}
		CHECK_EQ(bottom[0]->shape(0), bottom[1]->shape(0));
		CHECK_EQ(bottom[0]->shape(1), bottom[1]->shape(1));
	}

	template <typename Dtype>
	void SeqEndLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top){
		CHECK_EQ(bottom[1]->num_axes(), 2);
		CHECK_EQ(bottom[0]->num_axes(), 3);
		CHECK_EQ(bottom[0]->shape(0), bottom[1]->shape(0));
		CHECK_EQ(bottom[0]->shape(1), bottom[1]->shape(1));
		end_id_.clear();
		const Dtype* cont_data = bottom[1]->cpu_data();
		const int T = bottom[1]->shape(0);
		int cont_t;
		for (int t = 1; t < T; ++t){
			// NOTE: we just use cont of the first stream to infer sequence end
			// maybe a bug
			cont_t = static_cast<int>(*(cont_data + bottom[1]->offset(t)));
			if (cont_t == 0){
				end_id_.push_back(t - 1);
			}
		}
		end_id_.push_back(T - 1);
		vector<int> top_shape = bottom[0]->shape();
		top_shape[0] = end_id_.size();
		top[0]->Reshape(top_shape);
	}

	template <typename Dtype>
	void SeqEndLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top){
		const Dtype* bottom_data = bottom[0]->cpu_data();
		Dtype* top_data = top[0]->mutable_cpu_data();
		int num_seq = end_id_.size();
		int outer_dim = bottom[0]->count(1);
		const Dtype* bottom_offset;
		for (int i = 0; i < num_seq; ++i){
			bottom_offset = bottom_data + outer_dim * static_cast<int>(end_id_[i]);
			caffe_copy(outer_dim, bottom_offset, top_data);
			top_data += top[0]->offset(1);
		}
	}

	template <typename Dtype>
	void SeqEndLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom){
		if (propagate_down[0]){
			const Dtype* top_diff = top[0]->cpu_diff();
			Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
			int num_seq = end_id_.size();
			int outer_dim = bottom[0]->count(1);
			Dtype* bottom_offset;
			for (int i = 0; i < num_seq; ++i){
				bottom_offset = bottom_diff + outer_dim * static_cast<int>(end_id_[i]);
				caffe_copy(outer_dim, top_diff, bottom_offset);
				top_diff += top[0]->offset(1);
			}
		}
	}

#ifdef CPU_ONLY
	STUB_GPU(SeqEndLayer);
#endif

	INSTANTIATE_CLASS(SeqEndLayer);
	REGISTER_LAYER_CLASS(SeqEnd);

}// namespace caffe
