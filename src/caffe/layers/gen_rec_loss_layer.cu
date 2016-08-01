#include <vector>
#include <utility>

#include "caffe/layers/gen_rec_loss_layer.hpp"

namespace caffe{
	template <typename Dtype>
	void GenRecLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top){
		const Dtype* mu_data = bottom[0]->gpu_data();
		const Dtype* sigma_data = bottom[1]->gpu_data();
		const Dtype* x_data = bottom[2]->gpu_data();
		//put intemediate result of mu into buffer data
		Dtype* mu_buffer_data = mu_sigma_buffer_.mutable_gpu_data();
		//put intemediate result of sigma into buffer diff
		Dtype* sigma_buffer_data = mu_sigma_buffer_.mutable_gpu_diff();
		const int count = bottom[0]->count();
		caffe_gpu_sub(count, x_data, mu_data, mu_buffer_data);
		caffe_gpu_sqr(count, mu_buffer_data, mu_buffer_data);
		caffe_gpu_sqr(count, sigma_data, sigma_buffer_data);
		//put the div result into multiplier diff
		caffe_gpu_div(count, mu_buffer_data, sigma_buffer_data, sum_multiplier_.mutable_gpu_diff());
		//sum
		Dtype loss1;
		caffe_gpu_dot(count, sum_multiplier_.gpu_data(), sum_multiplier_.mutable_gpu_diff(), &loss1);
		//since the invert sqare of sigma is not needed any more, we can just put log\sigma into this 
		//memory again
		caffe_gpu_log(count, sigma_data, sigma_buffer_data);
		Dtype loss2;
		caffe_gpu_dot(count, sum_multiplier_.gpu_data(), sigma_buffer_data, &loss2);
		Dtype loss3 = Dtype(0.5) * Dtype(num_feats_) * log(2 * Dtype(PI));
		Dtype loss = loss1 + loss2 + loss3;
		top[0]->mutable_cpu_data()[0] = loss / bottom[0]->num();
	}

	template <typename Dtype>
	__global__ void genrec_loss_mu_backward_kernel(const int n, const Dtype coeff,
		const Dtype* mu_data, const Dtype* sigma_data, const Dtype* x_data, Dtype* mu_diff){
		CUDA_KERNEL_LOOP(index, n){
			mu_diff[index] = coeff * Dtype(-2) * (x_data[index] - mu_data[index]) /
				(sigma_data[index] * sigma_data[index]);
		}
	}

	template <typename Dtype>
	__global__ void genrec_loss_sigma_backward_kernel(const int n, const Dtype coeff,
		const Dtype* mu_data, const Dtype* sigma_data, const Dtype* x_data, Dtype* sigma_diff){
		CUDA_KERNEL_LOOP(index, n){
			sigma_diff[index] = coeff * Dtype(-2) * (x_data[index] - mu_data[index]) *
				(x_data[index] - mu_data[index]) / pow(sigma_data[index], Dtype(3)) +
				coeff / sigma_data[index];
		}
	}

	template <typename Dtype>
	void GenRecLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down,
		const vector<Blob<Dtype>*>& bottom){
		Dtype coeff = top[0]->cpu_diff()[0] / bottom[0]->num();
		const Dtype* mu_data = bottom[0]->gpu_data();
		const Dtype* sigma_data = bottom[1]->gpu_data();
		const Dtype* x_data = bottom[2]->gpu_data();
		Dtype* mu_diff = bottom[0]->mutable_gpu_diff();
		Dtype* sigma_diff = bottom[1]->mutable_gpu_diff();
		const int count = bottom[0]->count();
		if (propagate_down[0]){
			genrec_loss_mu_backward_kernel<Dtype> << <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS >> >(
				count, coeff, mu_data, sigma_data, x_data, mu_diff);
			CUDA_POST_KERNEL_CHECK;
		}
		if (propagate_down[1]){
			genrec_loss_sigma_backward_kernel<Dtype> << <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS >> >(
				count, coeff, mu_data, sigma_data, x_data, sigma_diff);
			CUDA_POST_KERNEL_CHECK;
		}
	}

	INSTANTIATE_LAYER_GPU_FUNCS(GenRecLossLayer);
} // namespace caffe