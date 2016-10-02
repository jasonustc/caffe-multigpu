#include <vector>

#include "caffe/layers/switch_layer.hpp"

namespace caffe{
	template <typename Dtype>
	void SwitchLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top){
		const unsigned int rng_seed = caffe_rng_rand();
		rng_.reset(new Caffe::RNG(rng_seed));
		vector<int> bottom_shape = bottom[0]->shape();
		has_bottom_index_ = this->layer_param_.switch_param().has_bottom_index();
		for (int n = 1; n < bottom.size(); ++n){
			if (has_bottom_index_ && n == (bottom.size() - 1)){
				continue;
			}
			for (size_t i = 0; i < bottom_shape.size(); ++i){
				CHECK_EQ(bottom_shape[i], bottom[n]->shape(i));
			}
		}
		// output bottom_index
		if (top.size() > 1){
			top[1]->Reshape(vector<int>(1, 1));
		}
	}
	/// Currently, only randomly switch is implemeted, 
	/// TODO: orderly switch
	template <typename Dtype>
	void SwitchLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top){
		top[0]->ReshapeLike(*bottom[0]);
	}

	template <typename Dtype>
	void SwitchLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top){
		if (has_bottom_index_){ // got input index
			bottom_index_ = bottom[bottom.size() - 1]->cpu_data()[0];
		}
		else{
			bottom_index_ = Rand(bottom.size());
			// for gtest 
//			bottom_index_ = 1;
		}
		caffe_copy(top[0]->count(), bottom[bottom_index_]->cpu_data(),
			top[0]->mutable_cpu_data());
		if (top.size() > 1){
			// output bottom index
			top[1]->mutable_cpu_data()[0] = bottom_index_;
		}
	}

	template <typename Dtype>
	void SwitchLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down,
		const vector<Blob<Dtype>*>& bottom){
		LOG(INFO) << "bottom_index_" << bottom_index_;
		if (propagate_down[bottom_index_]){
			caffe_copy(top[0]->count(),
				top[0]->cpu_diff(),
				bottom[bottom_index_]->mutable_cpu_diff());
		}
	}

#ifdef CPU_ONLY
	STUB_GPU(SwitchLayer);
#endif

	INSTANTIATE_CLASS(SwitchLayer);
	REGISTER_LAYER_CLASS(Switch);
} // namespace caffe