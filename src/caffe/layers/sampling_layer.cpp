#include <vector>
#include <utility>

#include "caffe/layers/sampling_layer.hpp"

namespace caffe{
	template <typename Dtype>
	void SamplingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top){
		Layer<Dtype>::LayerSetUp(bottom, top);
		CHECK(bottom[0]->shape() == bottom[1]->shape()) << "The shape of mu and sigma \
														   			should be the same";
	}

	template <typename Dtype>
	void SamplingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top){
		top[0]->ReshapeLike(*bottom[0]);
		gaussian_value_.ReshapeLike(*bottom[0]);
	}

	template <typename Dtype>
	void SamplingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top){
		const Dtype* mu_data = bottom[0]->cpu_data();
		const Dtype* sigma_data = bottom[1]->cpu_data();
		const int count = bottom[0]->count();
		Dtype* top_data = top[0]->mutable_cpu_data();
		Dtype* gaussian_data = gaussian_value_.mutable_cpu_data();
		caffe_rng_gaussian(count, Dtype(0), Dtype(1), gaussian_data);
		//z_t = \mu + \sigma * N(0,1)
		caffe_mul(count, sigma_data, gaussian_data, top_data);
		caffe_add(count, mu_data, top_data, top_data);
	}

	template <typename Dtype>
	void SamplingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down,
		const vector<Blob<Dtype>*>& bottom){
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

	INSTANTIATE_CLASS(SamplingLayer);
	REGISTER_LAYER_CLASS(Sampling);
} // namespace caffe
