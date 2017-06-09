#include <vector>

#include "caffe/layers/euclidean_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {
template <typename Dtype>
void EuclideanLossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*> & bottom,
	const vector<Blob<Dtype>*>& top){
	LossLayer<Dtype>::LayerSetUp(bottom, top);
	CHECK_LT(axis_, bottom[0]->num_axes());
	// setup scale layer
	if (bottom.size() > 2){
		// put all scale parameters in scale_param
		scale_layer_.reset(new ScaleLayer<Dtype>(this->layer_param_));
		vector<Blob<Dtype>*> scale_bottom(2);
		// use bottom[1] for shape reference
		scale_bottom[0] = bottom[1];
		scale_bottom[1] = bottom[2];
		// ScaleLayer allows in-place operation
		// so we use diff_ as both scale_bottom and scale_top
		vector<Blob<Dtype>*> scale_top(1, &diff_);
		scale_layer_->SetUp(scale_bottom, scale_top);
	}
}

template <typename Dtype>
void EuclideanLossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->count(1), bottom[1]->count(1))
      << "Inputs must have the same dimension.";
  diff_.ReshapeLike(*bottom[0]);
  if (bottom.size() > 2){
	  vector<Blob<Dtype>*> scale_bottom(2);
	  scale_bottom[0] = &diff_;
	  scale_bottom[1] = bottom[2];
	  // ScaleLayer allows in-place operation
	  // so we use diff_ as both scale_bottom and scale_top
	  vector<Blob<Dtype>*> scale_top(1, &diff_);
	  scale_layer_->Reshape(scale_bottom, scale_top);
  }
}

template <typename Dtype>
void EuclideanLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int count = bottom[0]->count();
  caffe_sub(
      count,
      bottom[0]->cpu_data(),
      bottom[1]->cpu_data(),
      diff_.mutable_cpu_data());
  if (bottom.size() > 2){
	  vector<Blob<Dtype>*> scale_bottom(2);
	  scale_bottom[0] = &diff_;
	  scale_bottom[1] = bottom[2];
	  // ScaleLayer allows in-place operation
	  // so we use diff_ as both scale_bottom and scale_top
	  vector<Blob<Dtype>*> scale_top(1, &diff_);
	  scale_layer_->Forward(scale_bottom, scale_top);
  }
  Dtype dot = caffe_cpu_dot(count, diff_.cpu_data(), diff_.cpu_data());
  Dtype loss = dot / bottom[0]->count(0, axis_) / Dtype(2);
  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void EuclideanLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  for (int i = 0; i < 2; ++i) {
	 // because corresponding feats in diff_ is set to 0 in the forward pass
	 // so here we do not need to do anything
    if (propagate_down[i]) {
      const Dtype sign = (i == 0) ? 1 : -1;
      const Dtype alpha = sign * top[0]->cpu_diff()[0] / bottom[i]->count(0, axis_);
      caffe_cpu_axpby(
          bottom[i]->count(),              // count
          alpha,                              // alpha
          diff_.cpu_data(),                   // a
          Dtype(0),                           // beta
          bottom[i]->mutable_cpu_diff());  // b
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(EuclideanLossLayer);
#endif

INSTANTIATE_CLASS(EuclideanLossLayer);
REGISTER_LAYER_CLASS(EuclideanLoss);

}  // namespace caffe
