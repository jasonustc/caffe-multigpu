#include <vector>
#include "caffe/layers/scalar_scale_layer.hpp"

namespace caffe{
	/*
	 * In-place operation is allowed in this layer.
	 */
	template <typename Dtype>
	void ScalarScaleLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top){
		NeuronLayer<Dtype>::LayerSetUp(bottom, top);
		scale_ = this->layer_param_.scale_param().scale();
	}

	template <typename Dtype>
	void ScalarScaleLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top){
		int count = bottom[0]->count();
		caffe_copy(count, bottom[0]->cpu_data(), top[0]->mutable_cpu_data());
		caffe_scal(count, scale_, top[0]->mutable_cpu_data());
	}

	template <typename Dtype>
	void ScalarScaleLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down,
		const vector<Blob<Dtype>*>& bottom){
		if (propagate_down[0]){
			int count = bottom[0]->count();
			caffe_copy(count, top[0]->cpu_diff(), bottom[0]->mutable_cpu_diff());
			caffe_scal(count, scale_, bottom[0]->mutable_cpu_diff());
		}
	}

#ifdef CPU_ONLY
	STUB_GPU(ScalarScaleLayer);
#endif

	INSTANTIATE_CLASS(ScalarScaleLayer);
	REGISTER_LAYER_CLASS(ScalarScale);

} // namespace caffe