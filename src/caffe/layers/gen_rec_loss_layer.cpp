#include <vector>
#include <utility>

#include "caffe/layers/gen_rec_loss_layer.hpp"

namespace caffe{
	template <typename Dtype>
	void GenRecLossLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top){
		CHECK(bottom[0]->shape() == bottom[1]->shape()) << "\mu and \sigma should have the "
  			<< " same shape.";
		CHECK(bottom[0]->shape() == bottom[2]->shape()) << "\mu and x should have the "
  			<< " same shape. " << "mu: " << bottom[0]->shape_string() << " x: " << bottom[2]->shape_string();
		//get number of features: H x W
		this->num_feats_ = bottom[0]->count(2);
		vector<int> top_shape(0);
		top[0]->Reshape(top_shape);
		sum_multiplier_.ReshapeLike(*bottom[0]);
		mu_sigma_buffer_.ReshapeLike(*bottom[0]);
		caffe_set(bottom[0]->count(), Dtype(1.), sum_multiplier_.mutable_cpu_data());
	}

	template <typename Dtype>
	void GenRecLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top){
		const Dtype* mu_data = bottom[0]->cpu_data();
		const Dtype* sigma_data = bottom[1]->cpu_data();
		const Dtype* x_data = bottom[2]->cpu_data();
		//put the intemediate result of sigma into buffer data
		Dtype* mu_buffer_data = mu_sigma_buffer_.mutable_cpu_data();
		//put the intemediate result of mu into buffer diff
		Dtype* sigma_buffer_data = mu_sigma_buffer_.mutable_cpu_diff();
		const int count = bottom[0]->count();
		caffe_sub(count, x_data, mu_data, mu_buffer_data);
		caffe_sqr(count, mu_buffer_data, mu_buffer_data);
		caffe_sqr(count, sigma_data, sigma_buffer_data);
		//put the result into sum_multiplier_ diff
		caffe_div(count, mu_buffer_data, sigma_buffer_data, sum_multiplier_.mutable_cpu_diff());
		//sum
		Dtype loss1 = caffe_cpu_dot(count, sum_multiplier_.cpu_diff(), sum_multiplier_.cpu_data());
		//use this memory again
		caffe_log(count, sigma_data, sigma_buffer_data);
		Dtype loss2 = caffe_cpu_dot(count, sigma_buffer_data, sum_multiplier_.cpu_data());
		//loss related to number of features
		//actually here should be the number of all dim of features
		Dtype loss3 = Dtype(0.5) * Dtype(num_feats_) * log(2 * Dtype(PI));
		Dtype loss = loss1 + loss2 + loss3;
		//averaged by the num of samples
		top[0]->mutable_cpu_data()[0] = loss / bottom[0]->num();
	}

	template <typename Dtype>
	void GenRecLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down,
		const vector<Blob<Dtype>*>& bottom){
		const int count = bottom[0]->count();
		const Dtype* mu_data = bottom[0]->cpu_data();
		const Dtype* sigma_data = bottom[1]->cpu_data();
		const Dtype* x_data = bottom[2]->cpu_data();
		Dtype* mu_diff = bottom[0]->mutable_cpu_diff();
		Dtype* sigma_diff = bottom[1]->mutable_cpu_diff();
		const Dtype coeff = top[0]->cpu_diff()[0] / bottom[0]->num();
		if (propagate_down[0]){
			for (int i = 0; i < count; i++){
				//2(x_i - \mu_i)/\sigma^2
				//avoid zero case in denomiter
				//because sigma = exp(W * h + b), maybe this will be not an issue
				Dtype sig = std::max(sigma_data[i], Dtype(kLOG_THRESHOLD));
				mu_diff[i] = Dtype(-2.) * (x_data[i] - mu_data[i]) * coeff / (sig * sig);
			}
		}
		if (propagate_down[1]){
			for (int i = 0; i < count; i++){
				//avoid zero case
				//because sigma = exp(W * h + b), maybe this will be not an issue
				Dtype sig = std::max(sigma_data[i], Dtype(kLOG_THRESHOLD));
				//-2(x_i - \mu_i)^2 / (\sigma_i^3) + 1/ \sigma_i
				sigma_diff[i] = coeff * Dtype(-2) * (x_data[i] - mu_data[i]) * (x_data[i] - mu_data[i]) /
					(sig * sig * sig) + coeff / sig;
			}
		}
	}

	INSTANTIATE_CLASS(GenRecLossLayer);
	REGISTER_LAYER_CLASS(GenRecLoss);
} // namespace caffe