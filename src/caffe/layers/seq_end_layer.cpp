#include <utility>
#include <vector>

#include "caffe/layers/seq_end_layer.hpp"

/// TODO: make this layer more compatible with dimensions
namespace caffe{
	template <typename Dtype>
	void SeqEndLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top){
		CHECK_EQ(bottom[0]->num_axes(), 3);
		CHECK_EQ(bottom[1]->num_axes(), 2);
		if (bottom[1]->shape(1) > 1){
			LOG(ERROR) << "Please make sure that each stream uses the same cont variable";
		}
		/// axis_ = bottom[0]->CanonicalAxisIndex(this->layer_param_.seq_end_param().axis());
		CHECK_EQ(bottom[0]->shape(0), bottom[1]->shape(0));
		CHECK_EQ(bottom[0]->shape(1), bottom[1]->shape(1));
		const int init_end_len = this->layer_param_.seq_end_param().init_end_len();
		CHECK_GE(init_end_len, 1);
		vector<int> top_shape = bottom[0]->shape();
		top_shape[0] = init_end_len;
		top[0]->Reshape(top_shape);
	}

	template <typename Dtype>
	void SeqEndLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top){
		CHECK_EQ(bottom[1]->num_axes(), 2);
		CHECK_EQ(bottom[0]->num_axes(), 3);
		CHECK_EQ(bottom[0]->shape(0), bottom[1]->shape(0));
		CHECK_EQ(bottom[0]->shape(1), bottom[1]->shape(1));
		// we do not put InferSeqEndId in here is because reshape is called 
		// in the set up stage, when all cont_t s are 0s, which may cause shape
		// inconsistent in following layers
	}

	template <typename Dtype>
	void SeqEndLayer<Dtype>::InferSeqEndId(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top){
		end_id_.clear();
		const Dtype* cont_data = bottom[1]->cpu_data();
		const int T = bottom[1]->shape(0);
		int cont_t;
		// NOTE: we just use cont of the first stream to infer sequence end
		// maybe a bug
		int seq_offset = bottom[1]->count(1);
		// skip the first element
		cont_data += seq_offset;
		for (int t = 1; t < T; ++t){
			cont_t = static_cast<int>(cont_data[0]);
			if (cont_t == 0){
				end_id_.push_back(t - 1);
			}
			cont_data += seq_offset;
		}
		end_id_.push_back(T - 1);
		vector<int> top_shape = bottom[0]->shape();
		top_shape[0] = end_id_.size();
		top[0]->Reshape(top_shape);
	}

	template <typename Dtype>
	void SeqEndLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top){
		this->InferSeqEndId(bottom, top);
		const Dtype* bottom_data = bottom[0]->cpu_data();
		Dtype* top_data = top[0]->mutable_cpu_data();
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
				top_diff += outer_dim;
			}
		}
	}

#ifdef CPU_ONLY
	STUB_GPU(SeqEndLayer);
#endif

	INSTANTIATE_CLASS(SeqEndLayer);
	REGISTER_LAYER_CLASS(SeqEnd);

}// namespace caffe
