#include <vector>

#include "caffe/layers/dropout_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void ClipData(const int n, const Dtype lower, const Dtype higher,
	Dtype* data){
	CUDA_KERNEL_LOOP(index, n){
		Dtype value = data[index];
		data[index] = value > higher ? higher : (value < lower ? lower : value);
	}
}

template <typename Dtype>
void DropoutLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int count = bottom[0]->count(0, num_axes_);
  Dtype* mask = rand_vec_->mutable_gpu_data();
  if (this->phase_ == TRAIN) {
	  switch (drop_type_){
	  case DropoutParameter_DropType_BERNOULLI:
	  {
	    // Create random numbers
	    caffe_gpu_rng_bernoulli(count, Dtype(1. - threshold_), mask);
	    break;
	  }
	  case DropoutParameter_DropType_GAUSSIAN:
	  {
	   caffe_gpu_rng_gaussian(count, Dtype(mu_), Dtype(sigma_), mask);
	   const int mask_count = rand_vec_->count();
	   // clip to be in [0,1]
	   ClipData<Dtype> // NOLINT_NEXT_LINE(whitespace/operators)
		   << <CAFFE_GET_BLOCKS(mask_count), CAFFE_CUDA_NUM_THREADS >> >
		   (count, Dtype(0), Dtype(1), mask);
		CUDA_POST_KERNEL_CHECK;
	   break;
	  }
	  case DropoutParameter_DropType_UNIFORM:
	  {
	    caffe_gpu_rng_uniform(count, Dtype(a_), Dtype(b_), mask);
		break;
	  }
	  }
	  vector<Blob<Dtype>*> scale_bottom(2, NULL);
	  scale_bottom[0] = bottom[0];
	  scale_bottom[1] = rand_vec_;
	  const vector<Blob<Dtype>*> scale_top(1, top[0]);
	  scale_layer_->Forward(scale_bottom, scale_top);
	  caffe_scal(top[0]->count(), scale_, top[0]->mutable_cpu_data());
  } else {
    caffe_copy(count, bottom_data, top_data);
  }
}

template <typename Dtype>
void DropoutLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    Dtype* top_diff = top[0]->mutable_gpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    if (this->phase_ == TRAIN) {
		// scale
		caffe_gpu_scal<Dtype>(top[0]->count(), scale_, top_diff);
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

INSTANTIATE_LAYER_GPU_FUNCS(DropoutLayer);

}  // namespace caffe
