// TODO (sergeyk): effect should not be dependent on phase. wasted memcpy.

#include <vector>

#include "caffe/layers/dropout_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void DropoutLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  NeuronLayer<Dtype>::LayerSetUp(bottom, top);
  threshold_ = this->layer_param_.dropout_param().dropout_ratio();
  DCHECK(threshold_ > 0.);
  DCHECK(threshold_ < 1.);
  scale_ = 1. / (1. - threshold_);
  uint_thres_ = static_cast<unsigned int>(UINT_MAX * threshold_);
  drop_type_ = this->layer_param_.dropout_param().drop_type();
  a_ = this->layer_param_.dropout_param().a();
  b_ = this->layer_param_.dropout_param().b();
  CHECK_LT(a_, b_);
  mu_ = this->layer_param_.dropout_param().mu();
  sigma_ = this->layer_param_.dropout_param().sigma();
  CHECK_GT(sigma_, 0);
  switch (drop_type_){
  case DropoutParameter_DropType_UNIFORM:
	  scale_ = 2. / (b_ + a_);
	  break;
  case DropoutParameter_DropType_GAUSSIAN:
	  scale_ = 1. / mu_;
	  break;
  case DropoutParameter_DropType_BERNOULLI:
	  scale_ = 1. / (1. - threshold_);
	  break;
  default:
	  LOG(FATAL) << "unknown dropout type";
  }
  // layer-wise dropout parameters
  layer_wise_ = this->layer_param_.dropout_param().layer_wise();
  axis_ = this->layer_param_.dropout_param().axis();
  CHECK_LT(axis_, bottom[0]->num_axes() - 1);
}

template <typename Dtype>
void DropoutLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  NeuronLayer<Dtype>::Reshape(bottom, top);
  // Set up the cache for random number generation
  // ReshapeLike does not work because rand_vec_ is of Dtype uint
  rand_vec_.ReshapeLike(*(bottom[0]));
}

template <typename Dtype>
void DropoutLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  Dtype* mask = rand_vec_.mutable_cpu_data();
  const int count = bottom[0]->count();
  if (this->phase_ == TRAIN) {
	  switch (drop_type_){
	  case DropoutParameter_DropType_BERNOULLI:
	  {
	    // Create random numbers
	    caffe_rng_bernoulli(count, 1. - threshold_, mask);
	    break;
	  }
	  case DropoutParameter_DropType_GAUSSIAN:
	  {
	   caffe_rng_gaussian(count, Dtype(mu_), Dtype(sigma_), mask);
	   // clip to be in [0,1]
	   for (int i = 0; i < count; ++i){
	  	 Dtype m = mask[i];
	  	 mask[i] = m > 1 ? 1 : (m < 0 ? 0 : m);
	   }
	   break;
	  }
	  case DropoutParameter_DropType_UNIFORM:
	  {
	    caffe_rng_uniform(count, a_, b_, mask);
		break;
	  }
	  }
    for (int i = 0; i < count; ++i) {
      top_data[i] = bottom_data[i] * mask[i] * scale_;
    }
  } else {
    caffe_copy(bottom[0]->count(), bottom_data, top_data);
  }
}

template <typename Dtype>
void DropoutLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    if (this->phase_ == TRAIN) {
      const unsigned int* mask = rand_vec_.cpu_data();
      const int count = bottom[0]->count();
      for (int i = 0; i < count; ++i) {
        bottom_diff[i] = top_diff[i] * mask[i] * scale_;
      }
    } else {
      caffe_copy(top[0]->count(), top_diff, bottom_diff);
    }
  }
}


#ifdef CPU_ONLY
STUB_GPU(DropoutLayer);
#endif

INSTANTIATE_CLASS(DropoutLayer);
REGISTER_LAYER_CLASS(Dropout);

}  // namespace caffe
