#include <vector>
#include <utility>

#include "caffe/layers/rbm_layer.hpp"

namespace caffe{
	template <typename Dtype>
	void RBMLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top){
		//top[0] shares data with pos_h_ data
		Gibbs_vhvh();
		//output reconstruction loss
		if (top.size() > 1){
			const int count = bottom[0]->count();
			const Dtype* bottom_data = bottom[0]->gpu_data();
			const Dtype* neg_v_data = neg_v_->gpu_data();
			Dtype* tmp_data = neg_v_->mutable_gpu_diff();
			caffe_gpu_sub<Dtype>(count, bottom_data, neg_v_data, tmp_data);
			Dtype loss;
			caffe_gpu_dot<Dtype>(count, tmp_data, tmp_data, &loss);
			top[1]->mutable_cpu_data()[0] = loss / bottom[0]->num();
		}
	}

	// TODO: maybe we can put rbm update in forward pass
	// and discriminative update in backward pass
	template <typename Dtype>
	void RBMLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down,
		const vector<Blob<Dtype>*>& bottom){
		//put positive data into buf data
		Dtype* pos_ass_data = weight_diff_buf_->mutable_gpu_data();
		//put negative data into buf diff
		Dtype* neg_ass_data = weight_diff_buf_->mutable_gpu_diff();
		const Dtype* pos_v_data = bottom[0]->gpu_data();
		const Dtype* pos_h_data = pos_h_->gpu_data();
		const Dtype* neg_v_data = neg_v_->gpu_data();
		const Dtype* neg_h_data = neg_h_->gpu_data();
		Dtype* weight_diff = this->blobs_[0]->mutable_gpu_diff();
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
		//\delta c_j = \delta c_j + p_h_j^(0) - p_h_j^(k)
		if (bias_term_ && this->param_propagate_down_[1]){
			const int count_h = pos_h_->count();
			Dtype* h_bias_diff = this->blobs_[1]->mutable_gpu_diff();
			//put buffer data in neg_h_.diff()
			//pos_h_ is shared with top[0], be carefully to use it in other place
			caffe_gpu_sub<Dtype>(count_h, pos_h_data, neg_h_data, neg_h_->mutable_gpu_diff());
			//put intemediate result into neg_h_ data
			//average by batch size
			caffe_gpu_gemv<Dtype>(CblasTrans, M_, N_, scale, neg_h_->gpu_diff(),
				bias_multiplier_->gpu_data(), (Dtype)1., h_bias_diff);
		}

		//Gradient with respect to v_bias
		//\delta b_j = \delta b_j + v_j^(0) - v_j^(k)
		if (bias_term_ && this->param_propagate_down_[2]){
			const int count_v = pos_v_->count();
			Dtype* v_bias_diff = this->blobs_[2]->mutable_gpu_diff();
			//put buffer data in neg_v_.diff()
			//pos_v_ is shared with bottom[0], be carefully to use it in other place
			caffe_gpu_sub<Dtype>(count_v, pos_v_data, neg_v_data, neg_v_->mutable_gpu_diff());
			//put intemediate result into neg_v_ data
			//average by batch size
			caffe_gpu_gemv<Dtype>(CblasTrans, M_, K_, scale, neg_v_->gpu_diff(),
				bias_multiplier_->gpu_data(), (Dtype)1., v_bias_diff);
		}

		if (propagate_down[0]){
			// sigmoid activation
			const vector<Blob<Dtype>*> act_bottom(1, top[0]);
			const vector<Blob<Dtype>*> act_top(1, top[0]);
			act_layer_->Backward(act_top, vector<bool>(1, true), act_bottom);
			// forward inner_product
			const vector<Blob<Dtype>*> ip_forward_bottom(1, bottom[0]);
			const vector<Blob<Dtype>*> ip_forward_top(1, top[0]);
			// TODO: another option is change the weights in this process
			ip_forward_layer_->set_param_propagate_down(0, false);
			ip_forward_layer_->set_param_propagate_down(1, false);
			ip_forward_layer_->Backward(ip_forward_top, vector<bool>(1, true), ip_forward_bottom);
		}
	}

	INSTANTIATE_LAYER_GPU_FUNCS(RBMLayer);
} // namespace caffe
