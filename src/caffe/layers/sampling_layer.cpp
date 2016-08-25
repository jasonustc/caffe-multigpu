#include <vector>
#include <utility>

#include "caffe/layers/sampling_layer.hpp"

namespace caffe{
	template <typename Dtype>
	void SamplingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top){
		Layer<Dtype>::LayerSetUp(bottom, top);
		CHECK(bottom[0]->shape() == bottom[1]->shape()) << 
			"The shape of mu and sigma  should be the same";
		sample_type_ = this->layer_param_.sampling_param().sample_type();
		// currently, backward is only allowed in gaussian sampling
		switch (sample_type_){
		case SamplingParameter_SampleType_GAUSSIAN:
		{
		  is_gaussian_ = true;
		  CHECK_EQ(bottom.size(), 2) << "two input blobs (\mu, \sigma) are required";
		  break;
		}
		case SamplingParameter_SampleType_UNIFORM:
		{
		   is_gaussian_ = false;
		   CHECK_EQ(bottom.size(), 2) << "two input blobs (a, b) are required";
		   break;
		}
		case SamplingParameter_SampleType_BERNOULLI:
		{
		   is_gaussian_ = false;
		   CHECK_EQ(bottom.size(), 1) << "only one blob is required (p)";
		   break;
		}
		default:
		{
           LOG(FATAL) << "Unkown sample type";
		}
		}
	}

	template <typename Dtype>
	void SamplingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top){
		if (bottom.size() == 2){
			CHECK_EQ(bottom[0]->count(), bottom[1]->count()) <<
				"bottom[0] and bottom[1] dimension not match";
		}
		top[0]->ReshapeLike(*bottom[0]);
		if (is_gaussian_){
			gaussian_value_.ReshapeLike(*bottom[0]);
		}
	}

	template <typename Dtype>
	void SamplingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top){
		Dtype* top_data = top[0]->mutable_cpu_data();
		const int count = bottom[0]->count();
		switch (sample_type_){
		case SamplingParameter_SampleType_GAUSSIAN:
		{
		  const Dtype* mu_data = bottom[0]->cpu_data();
		  const Dtype* sigma_data = bottom[1]->cpu_data();
		  Dtype* gaussian_data = gaussian_value_.mutable_cpu_data();
		  caffe_rng_gaussian(count, Dtype(0), Dtype(1), gaussian_data);
		  //z_t = \mu + \sigma * N(0,1)
		  caffe_mul(count, sigma_data, gaussian_data, top_data);
		  caffe_add(count, mu_data, top_data, top_data);
		  break;
		}
		case SamplingParameter_SampleType_UNIFORM:
		{
		 const Dtype* a_data = bottom[0]->cpu_data();
		 const Dtype* b_data = bottom[1]->cpu_data();
		 caffe_rng_uniform(count, a_data, b_data, top_data);
		 break;
		}
		case SamplingParameter_SampleType_BERNOULLI:
		{
		   const Dtype* p_data = bottom[0]->cpu_data();
		   caffe_rng_bernoulli(count, p_data, top_data);
		   break;
		}
		}
	}

	template <typename Dtype>
	void SamplingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down,
		const vector<Blob<Dtype>*>& bottom){
		if (is_gaussian_){
			const Dtype* top_diff = top[0]->cpu_diff();
			Dtype* mu_diff = bottom[0]->mutable_cpu_diff();
			Dtype* sigma_diff = bottom[1]->mutable_cpu_diff();
			const Dtype* gaussian_data = gaussian_value_.cpu_data();
			const int count = bottom[0]->count();
			if (propagate_down[0]){
				caffe_copy(count, top_diff, mu_diff);
			}
			if (propagate_down[1]){
				caffe_mul(count, top_diff, gaussian_data, sigma_diff);
			}
		}
		else{
			LOG(FATAL) << "backward is not implemented for bernoulli and uniform sampling";
		}
	}

	INSTANTIATE_CLASS(SamplingLayer);
	REGISTER_LAYER_CLASS(Sampling);
} // namespace caffe
