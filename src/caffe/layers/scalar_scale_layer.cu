#include <vector>
#include "caffe/layers/scalar_scale_layer.hpp"

namespace caffe{

	template <typename Dtype>
	void ScalarScaleLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top){
		int count = bottom[0]->count();
		caffe_copy(count, bottom[0]->gpu_data(), top[0]->mutable_gpu_data());
		caffe_gpu_scal(count, scale_, top[0]->mutable_gpu_data());
		caffe_gpu_add_scalar(count, bias_, top[0]->mutable_gpu_data());
	}

	template <typename Dtype>
	void ScalarScaleLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down,
		const vector<Blob<Dtype>*>& bottom){
		if (propagate_down[0]){
			int count = bottom[0]->count();
			caffe_copy(count, top[0]->gpu_diff(), bottom[0]->mutable_gpu_diff());
			caffe_gpu_scal(count, scale_, bottom[0]->mutable_gpu_diff());
		}
	}

	INSTANTIATE_LAYER_GPU_FUNCS(ScalarScaleLayer);

} // namespace caffe
