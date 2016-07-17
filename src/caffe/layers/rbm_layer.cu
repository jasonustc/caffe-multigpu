#include <vector>
#include <utility>

#include "caffe/layers/rbm_layer.hpp"

namespace caffe{

	template <typename Dtype>
	__global__ void SigmoidForward(const int n, const Dtype* in, Dtype* out){
		CUDA_KERNEL_LOOP(index, n){
			out[index] = 1. / (1. + exp(-in[index]));
		}
	}

	template <typename Dtype>
	void RBMLayer<Dtype>::Gibbs_vhvh_gpu(){
		const Dtype* weight_data = this->blobs_[0]->gpu_data();
		const Dtype* h_bias_data = this->blobs_[1]->gpu_data();
		const Dtype* v_bias_data = this->blobs_[2]->gpu_data();
		Dtype* pos_h_data = pos_h_.mutable_gpu_data();
		Dtype* neg_h_data = neg_h_.mutable_gpu_data();
		Dtype* positive_state_h_data = positive_state_h_.mutable_gpu_data();
		Dtype* negative_state_v_data = negative_state_v_.mutable_gpu_data();
		const Dtype* pos_v_data = pos_v_.gpu_data();
		Dtype* neg_v_data = neg_v_.mutable_gpu_data();
		const int count_h = pos_h_.count();
		const int count_v = neg_v_.count();
		//prop up
		//h: M x N  v: M x K w: N x K
		caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, N_, K_, (Dtype)1.,
			pos_v_data, weight_data, (Dtype)0, pos_h_data);
		if (bias_term_){
			caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, 1, (Dtype)1.,
				bias_multiplier_.gpu_data(), h_bias_data, (Dtype)1., pos_h_data);
		}
		//sigmoid activation
		SigmoidForward<Dtype> << <CAFFE_GET_BLOCKS(count_h), CAFFE_CUDA_NUM_THREADS >> >(
			count_h, pos_h_data, pos_h_data);
		//sampling
		caffe_gpu_rng_bernoulli<Dtype>(count_h, pos_h_data, positive_state_h_data);
		//prop down
		//h: M x N  v: M x K w: N x K
		//TODO: need to convert the data type of state_h to Dtype
		caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, K_, N_, (Dtype)1.,
			positive_state_h_data, weight_data, (Dtype)0., neg_v_data);
		if (bias_term_){
			caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, K_, 1, (Dtype)1.,
				bias_multiplier_.gpu_data(), v_bias_data, (Dtype)1., neg_v_data);
		}
		//sigmoid activation
		SigmoidForward<Dtype> << <CAFFE_GET_BLOCKS(count_v), CAFFE_CUDA_NUM_THREADS >> >(
			count_v, neg_v_data, neg_v_data);
		//sampling 
		caffe_gpu_rng_bernoulli<Dtype>(count_v, neg_v_data, negative_state_v_data);

		//prop up again
		caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, N_, K_, (Dtype)1.,
			negative_state_v_data, weight_data, (Dtype)0, neg_h_data);
		if (bias_term_){
			caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, 1, (Dtype)1.,
				bias_multiplier_.gpu_data(), h_bias_data, (Dtype)1., neg_h_data);
		}

		//sigmoid activation
		SigmoidForward<Dtype> << <CAFFE_GET_BLOCKS(count_h), CAFFE_CUDA_NUM_THREADS >> >(
			count_h, neg_h_data, neg_h_data);
	}

	template <typename Dtype>
	void RBMLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top){
		//top[0] shares data with pos_h_ data
		Gibbs_vhvh_gpu();
		//output reconstruction loss
		if (top.size() > 1){
			const int count = bottom[0]->count();
			const Dtype* bottom_data = bottom[0]->gpu_data();
			const Dtype* neg_v_data = neg_v_.gpu_data();
			Dtype* tmp_data = neg_v_.mutable_gpu_diff();
			caffe_gpu_sub<Dtype>(count, bottom_data, neg_v_data, tmp_data);
			Dtype loss;
			caffe_gpu_dot<Dtype>(count, tmp_data, tmp_data, &loss);
			top[1]->mutable_cpu_data()[0] = loss / bottom[0]->num();
		}
	}

	template <typename Dtype>
	void RBMLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down,
		const vector<Blob<Dtype>*>& bottom){
		//put positive data into buf data
		Dtype* pos_ass_data = weight_diff_buf_.mutable_gpu_data();
		//put negative data into buf diff
		Dtype* neg_ass_data = weight_diff_buf_.mutable_gpu_diff();
		const Dtype* pos_v_data = bottom[0]->gpu_data();
		const Dtype* pos_h_data = pos_h_.gpu_data();
		const Dtype* neg_v_data = neg_v_.gpu_data();
		const Dtype* neg_h_data = neg_h_.gpu_data();
		const Dtype* top_diff = top[0]->gpu_diff();
		Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
		Dtype* weight_diff = this->blobs_[0]->mutable_gpu_diff();
		const Dtype* weight_data = this->blobs_[0]->gpu_data();
		Dtype scale = Dtype(1.) / bottom[0]->num();

		//Gradient with respect to weight
		if (this->param_propagate_down_[0]){
			caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, N_, K_, M_, (Dtype)1.,
				pos_h_data, pos_v_data, (Dtype)0., pos_ass_data);
			caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, N_, K_, M_, (Dtype)1.,
				neg_h_data, neg_v_data, (Dtype)0., neg_ass_data);
			caffe_gpu_sub(N_ * K_, pos_ass_data, neg_ass_data, neg_ass_data);
			//average by batch size
			caffe_gpu_axpby<Dtype>(this->blobs_[0]->count(), scale, neg_ass_data,
				Dtype(1.), weight_diff);
		}

		//Gradient with respect to h_bias
		const int count_h = pos_h_.count();
		Dtype* h_bias_diff = this->blobs_[1]->mutable_gpu_diff();
		//\delta c_j = \delta c_j + p_h_j^(0) - p_h_j^(k)
		if (this->param_propagate_down_[1]){
			//put buffer data in neg_h_.diff()
			//pos_h_ is shared with top[0], be carefully to use it in other place
			caffe_gpu_sub<Dtype>(count_h, pos_h_data, neg_h_data, neg_h_.mutable_gpu_diff());
			//put intemediate result into neg_h_ data
			//average by batch size
			caffe_gpu_gemv<Dtype>(CblasTrans, M_, N_, scale, neg_h_.gpu_diff(),
				bias_multiplier_.gpu_data(), (Dtype)1., h_bias_diff);
		}

		//Gradient with respect to v_bias
		const int count_v = pos_v_.count();
		Dtype* v_bias_diff = this->blobs_[2]->mutable_gpu_diff();
		//\delta b_j = \delta b_j + v_j^(0) - v_j^(k)
		if (this->param_propagate_down_[2]){
			//put buffer data in neg_v_.diff()
			//pos_v_ is shared with bottom[0], be carefully to use it in other place
			caffe_gpu_sub<Dtype>(count_v, pos_v_data, neg_v_data, neg_v_.mutable_gpu_diff());
			//put intemediate result into neg_v_ data
			//average by batch size
			caffe_gpu_gemv<Dtype>(CblasTrans, M_, K_, scale, neg_v_.gpu_diff(),
				bias_multiplier_.gpu_data(), (Dtype)1., v_bias_diff);
		}

		if (propagate_down[0]){
			caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, K_, N_, (Dtype)1.,
				top_diff, weight_data, (Dtype)0., bottom_diff);
		}
		LOG(INFO) << "sqr_diff: " << this->blobs_[0]->sumsq_diff();
		LOG(INFO) << "abs_diff: " << this->blobs_[0]->asum_diff();
		LOG(INFO) << "abs_data: " << this->blobs_[0]->asum_data();
		LOG(INFO) << "sqr_data: " << this->blobs_[0]->sumsq_data();
		LOG(INFO) << "sqr_top: " << top[0]->sumsq_data();
		LOG(INFO) << "abs_top: " << top[0]->asum_data();
		LOG(INFO) << "abs_top_diff: " << top[0]->asum_diff();
		LOG(INFO) << "sqr_top_diff: " << top[0]->sumsq_diff();
	}

	INSTANTIATE_LAYER_GPU_FUNCS(RBMLayer);
} // namespace caffe