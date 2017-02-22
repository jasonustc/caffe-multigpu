#include <vector>

#include "caffe/layers/broadcast_layer.hpp"

namespace caffe{
	template <typename Dtype>
	void BroadcastLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top){
		const BroadcastParameter& param = this->layer_param_.broadcast_param();
		axis_ = bottom[0]->CanonicalAxisIndex(param.axis());
	}

	template <typename Dtype>
	void BroadcastLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top){
		int Ssize = bottom[0]->shape(axis_);
		int Tsize = bottom[1]->shape(axis_);
		int n = Tsize / Ssize;
		CHECK_EQ(n * Ssize, Tsize) << "The dimension of bottom[1] in axis "
			<< axis_ << " should be divided by that of bottom[0]";
		for (int i = 0; i < axis_; ++i){
			CHECK_EQ(bottom[0]->shape(i), bottom[1]->shape(i))
				<< "dimension not match for axis " << i;
		}
		vector<int> shape = bottom[0]->shape();
		shape[axis_] = Tsize;
		top[0]->Reshape(shape);
	}
	
	template <typename Dtype>
	void BroadcastLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top){
		int Ssize = bottom[0]->shape(axis_);
		int Tsize = bottom[1]->shape(axis_);
		int n = Tsize / Ssize;
		int num = bottom[0]->count(0, axis_);
		int count = bottom[0]->count(axis_);
		const Dtype* source_data = bottom[0]->cpu_data();
		Dtype* target_data = top[0]->mutable_cpu_data();
		for (int k = 0; k < num; ++k){
			for (int i = 0; i < n; ++i){
				caffe_copy(count, source_data, target_data);
				target_data += count;
			}
			source_data += count;
		}
	}

	template <typename Dtype>
	void BroadcastLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down,
		const vector<Blob<Dtype>*>& bottom){
		if (propagate_down[0]){
			int Ssize = bottom[0]->shape(axis_);
			int Tsize = bottom[1]->shape(axis_);
			int n = Tsize / Ssize;
			int num = bottom[0]->count(0, axis_);
			int count = bottom[0]->count(axis_);
			const Dtype* top_diff = top[0]->cpu_diff();
			Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
			for (int k = 0; k < num; ++k){
				// reset first
				caffe_copy(count, top_diff, bottom_diff);
				top_diff += count;
				for (int i = 1; i < n; ++i){
					caffe_axpy(count, Dtype(1.), top_diff, bottom_diff);
					top_diff += count;
				}
				bottom_diff += count;
			}
		}
	}

#ifdef CPU_ONLY
STUB_GPU(BroadcastLayer);
#endif

INSTANTIATE_CLASS(BroadcastLayer);
REGISTER_LAYER_CLASS(Broadcast);
} // namespace caffe