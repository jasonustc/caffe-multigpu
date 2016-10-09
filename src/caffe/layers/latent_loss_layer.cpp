#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/layer_factory.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/latent_loss_layer.hpp"

namespace caffe{
	template <typename Dtype>
	void LatentLossLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top){
		CHECK(bottom[0]->shape() == bottom[1]->shape()) << "mu and sigma must \
														   			have the same shape";
		//loss layer output a scalar, 0 shape
		vector<int> shape(0);
		top[0]->Reshape(shape);
		log_square_sigma_.ReshapeLike(*bottom[1]);
		sum_multiplier_.ReshapeLike(*bottom[1]);
		caffe_set(bottom[1]->count(), Dtype(1.), sum_multiplier_.mutable_cpu_data());
	}

	template <typename Dtype>
	void LatentLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top){
		const Dtype* mu_data = bottom[0]->cpu_data();
		const Dtype* sigma_data = bottom[1]->cpu_data();
		const int count = bottom[0]->count();
		Dtype* log_sqr_sigma_data = log_square_sigma_.mutable_cpu_data();
		caffe_sqr(count, sigma_data, log_sqr_sigma_data);
		caffe_log(count, log_sqr_sigma_data, log_sqr_sigma_data);
		Dtype dot_sigma = caffe_cpu_dot(count, sigma_data, sigma_data);
		Dtype dot_mu = caffe_cpu_dot(count, mu_data, mu_data);
		//dot with ones blob with same size to get the sum of this blob
		Dtype sum_log_sqr_sigma = caffe_cpu_dot(count, log_sqr_sigma_data, 
			sum_multiplier_.cpu_data());
		//\sum \sigma^2 + \mu^2 + log\sigma^2 - 1
		Dtype loss = (dot_sigma + dot_mu - sum_log_sqr_sigma) / bottom[0]->num() / Dtype(2);
		loss -= bottom[0]->count() / bottom[0]->num() / Dtype(2);
		top[0]->mutable_cpu_data()[0] = loss;
	}

	template <typename Dtype>
	void LatentLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down,
		const vector<Blob<Dtype>*>& bottom){
		const int count = bottom[0]->count();
		const Dtype* sigma_data = bottom[1]->cpu_data();
		Dtype* sigma_diff = bottom[1]->mutable_cpu_diff();
		const Dtype alpha = top[0]->cpu_diff()[0] / bottom[0]->num();
		if (propagate_down[0]){
			caffe_cpu_axpby(count, alpha, bottom[0]->cpu_data(), Dtype(0), 
				bottom[0]->mutable_cpu_diff());
		}
		if (propagate_down[1]){
			for (int i = 0; i < bottom[1]->count(); i++){
				Dtype sig = std::max(sigma_data[i], Dtype(kLOG_THRESHOLD));
				sig -= 1 / sig;
				sigma_diff[i] = alpha * sig;
			}
		}
	}

#ifdef  CPU_ONLY
	STUB_GPU(LatentLossLayer);
#endif

	INSTANTIATE_CLASS(LatentLossLayer);
	REGISTER_LAYER_CLASS(LatentLoss);
}