#include <vector>

#include "caffe/layers/euclidean_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void EuclideanLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int count = bottom[0]->count();
  caffe_gpu_sub(
      count,
      bottom[0]->gpu_data(),
      bottom[1]->gpu_data(),
      diff_.mutable_gpu_data());
  Dtype dot;
  caffe_gpu_dot(count, diff_.gpu_data(), diff_.gpu_data(), &dot);
  Dtype loss = dot / bottom[0]->num() / Dtype(2);
  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void EuclideanLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	if (iter_idx_ % update_iter_ == 0){
		for (int i = 0; i < 2; ++i) {
			if (propagate_down[i]) {
				const Dtype sign = (i == 0) ? 1 : -1;
				const Dtype alpha = sign * top[0]->cpu_diff()[0] / bottom[i]->num();
				caffe_gpu_axpby(
					bottom[i]->count(),              // count
					alpha,                              // alpha
					diff_.gpu_data(),                   // a
					Dtype(0),                           // beta
					bottom[i]->mutable_gpu_diff());  // b
			}
		}
	}
	else{
		for (int i = 0; i < 2; ++i){
			caffe_gpu_set(bottom[i]->count(), Dtype(0.), bottom[i]->mutable_gpu_diff());
		}
	}
	iter_idx_ = iter_idx_ == update_iter_ ? 1 : iter_idx_ + 1;
}

INSTANTIATE_LAYER_GPU_FUNCS(EuclideanLossLayer);

}  // namespace caffe
