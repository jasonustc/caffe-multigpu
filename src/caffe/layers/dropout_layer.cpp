// TODO (sergeyk): effect should not be dependent on phase. wasted memcpy.

#include <vector>

#include "caffe/layers/dropout_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void DropoutLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  NeuronLayer<Dtype>::LayerSetUp(bottom, top);
  const DropoutParameter& param = this->layer_param_.dropout_param();
  threshold_ = param.dropout_ratio();
  DCHECK(threshold_ > 0.);
  DCHECK(threshold_ < 1.);
  scale_ = 1. / (1. - threshold_);
  drop_type_ = param.drop_type();
  a_ = param.a();
  b_ = param.b();
  CHECK_LT(a_, b_);
  mu_ = param.mu();
  sigma_ = param.sigma();
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
  // layer-wise or element wise dropout parameters
  num_axes_ = param.num_axes() == -1 ? bottom[0]->num_axes() : param.num_axes();
  CHECK_LE(num_axes_, bottom[0]->num_axes());
  CHECK_GE(num_axes_, 1);
  vector<int> mask_shape = bottom[0]->shape();
  mask_shape.resize(num_axes_);
  //only need [0, ..., axis_] mask variables
  rand_vec_ = new Blob<Dtype>(mask_shape);
  LayerParameter scale_param;
  scale_param.mutable_scale_param()->set_axis(0);
  scale_param.mutable_scale_param()->set_num_axes(num_axes_);
  scale_layer_.reset(new ScaleLayer<Dtype>(scale_param));
  vector<Blob<Dtype>*> scale_bottom(2, NULL);
  scale_bottom[0] = bottom[0];
  scale_bottom[1] = rand_vec_;
  const vector<Blob<Dtype>*> scale_top(1, top[0]);
  scale_layer_->SetUp(scale_bottom, scale_top);
}

template <typename Dtype>
void DropoutLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  NeuronLayer<Dtype>::Reshape(bottom, top);
  vector<int> mask_shape = bottom[0]->shape();
  mask_shape.resize(num_axes_);
  rand_vec_->Reshape(mask_shape);
}

template <typename Dtype>
void DropoutLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  Dtype* mask = rand_vec_->mutable_cpu_data();
  const int count = bottom[0]->count(0, num_axes_);
  if (this->phase_ == TRAIN) {
	  switch (drop_type_){
	  case DropoutParameter_DropType_BERNOULLI:
	  {
	    // Create random numbers
	    caffe_rng_bernoulli(count, Dtype(1. - threshold_), mask);
	    break;
	  }
	  case DropoutParameter_DropType_GAUSSIAN:
	  {
	   caffe_rng_gaussian(count, Dtype(mu_), Dtype(sigma_), mask);
	   // clip to be in [0,1]
	   for (int i = 0; i < rand_vec_->count(); ++i){
	  	 Dtype m = mask[i];
	  	 mask[i] = m > 1 ? 1 : (m < 0 ? 0 : m);
	   }
	   break;
	  }
	  case DropoutParameter_DropType_UNIFORM:
	  {
	    caffe_rng_uniform(count, Dtype(a_), Dtype(b_), mask);
		break;
	  }
	  }
	  caffe_set<Dtype>(count, 1, mask);
	  vector<Blob<Dtype>*> scale_bottom(2, NULL);
	  scale_bottom[0] = bottom[0];
	  scale_bottom[1] = rand_vec_;
	  const vector<Blob<Dtype>*> scale_top(1, top[0]);
	  scale_layer_->Forward(scale_bottom, scale_top);
	  caffe_scal(top[0]->count(), scale_, top_data);
  } else {
    caffe_copy(bottom[0]->count(), bottom_data, top_data);
  }
}

template <typename Dtype>
void DropoutLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    Dtype* top_diff = top[0]->mutable_cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    if (this->phase_ == TRAIN) {
		// scale
		caffe_scal(top[0]->count(), scale_, top_diff);
		// multiply mask
		vector<Blob<Dtype>*> scale_bottom(2, NULL);
		scale_bottom[0] = bottom[0];
		scale_bottom[1] = rand_vec_;
		const vector<Blob<Dtype>*> scale_top(1, top[0]);
		vector<bool> prop_down(2, true);
		prop_down[1] = false;
		scale_layer_->Backward(scale_top, prop_down, scale_bottom);
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
