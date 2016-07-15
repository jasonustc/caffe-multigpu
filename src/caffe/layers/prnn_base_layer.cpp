#include <utility>
#include <string>
#include <vector>

#include "caffe/layers/prnn_base_layer.hpp"

namespace caffe{
	template <typename Dtype>
	void PRNNBaseLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top){
		CHECK_EQ(3, bottom[0]->num_axes());
		output_dim_ = this->layer_param_.recurrent_param().output_dim();
		L_ = this->layer_param_.recurrent_param().pred_length();
		const int N = bottom[0]->shape(0);
		T_ = N * L_;
		hidden_dim_ = bottom[0]->shape(2);
		// h_shape
		vector<int> h_shape(3, 1);
		h_shape[1] = bottom[0]->shape(1);
		h_shape[2] = hidden_dim_;
		// setup slice_h0_ layer
		// Top
		H0_.resize(N);
		for (int n = 0; n < N; ++n){
			H0_[n].reset(new Blob<Dtype>(h_shape));
		}
		// Layer
		const vector<Blob<Dtype>*> slice_bottom(1, bottom[0]);
		const vector<Blob<Dtype>*> slice_top(N, H0_[0].get());
		LayerParameter slice_param;
		slice_param.mutable_slice_param()->set_axis(0);
		slice_h0_.reset(new SliceLayer<Dtype>(slice_param));
		slice_h0_->SetUp(slice_bottom, slice_top);

		// setup ip_h_ layer
		// Bottom && Top
		vector<int> y_shape(3, 1);
		y_shape[1] = bottom[0]->shape(1);
		y_shape[2] = output_dim_;
		start_blob_.reset(new Blob<Dtype>(y_shape));
		Y_.resize(T_);
		H_.resize(T_);
		for (int t = 0; t < T_; ++t){
			Y_[t].reset(new Blob<Dtype>(y_shape));
			H_[t].reset(new Blob<Dtype>(h_shape));
		}
		// Layer
		const vector<Blob<Dtype>*> ip_h_bottom(1, H_[0].get());
		const vector<Blob<Dtype>*> ip_h_top(1, Y_[0].get());
		LayerParameter ip_h_param(this->layer_param_);
		ip_h_param.mutable_inner_product_param()->set_num_output(output_dim_);
		ip_h_param.mutable_inner_product_param()->set_axis(2);
		ip_h_.reset(new InnerProductLayer<Dtype>(ip_h_param));
		ip_h_->SetUp(ip_h_bottom, ip_h_top);

		// setup concat_y_ layer
		// Bottom && Top
		vector<Blob<Dtype>*> concat_y_bottom(T_, NULL);
		for (int t = 0; t < T_; ++t){
			concat_y_bottom[t] = Y_[t].get();
		}
		const vector<Blob<Dtype>*> concat_y_top(1, top[0]);
		// Layer
		LayerParameter concat_param;
		concat_param.mutable_concat_param()->set_axis(0);
		concat_y_.reset(new ConcatLayer<Dtype>(concat_param));
		concat_y_->SetUp(concat_y_bottom, concat_y_top);
	}

	template <typename Dtype>
	void PRNNBaseLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top){
		CHECK_EQ(bottom[0]->num_axes(), 3);
		CHECK_EQ(bottom[0]->shape(2), hidden_dim_)
			<< "H0_ feat dim incompatible with plstm parameters.";
		const int N = bottom[0]->shape(0);
		const int T = N * L_;
		if (T != T_){
			T_ = T;
			vector<int> h_shape(3, 1);
			h_shape[1] = bottom[0]->shape(1);
			h_shape[2] = hidden_dim_;

			// reshape slice_h0_
			H0_.resize(N);
			for (int n = 0; n < N; ++n){
				H0_[n].reset(new Blob<Dtype>(h_shape));
			}
			const vector<Blob<Dtype>*> slice_bottom(1, bottom[0]);
			const vector<Blob<Dtype>*> slice_top(N, H0_[0].get());
			slice_h0_->Reshape(slice_bottom, slice_top);

			// reshape Bottom && Top for ip_h_
			vector<int> y_shape(3, 1);
			y_shape[1] = bottom[0]->shape(1);
			y_shape[2] = output_dim_;
			Y_.resize(T_);
			H_.resize(T_);
			for (int t = 0; t < T_; ++t){
				Y_[t].reset(new Blob<Dtype>(y_shape));
				H_[t].reset(new Blob<Dtype>(h_shape));
			}

			// reshape concat_y_
			vector<Blob<Dtype>*> concat_y_bottom(T_, NULL);
			for (int t = 0; t < T_; ++t){
				concat_y_bottom[t] = Y_[t].get();
			}
			const vector<Blob<Dtype>*> concat_y_top(1, top[0]);
			concat_y_->Reshape(concat_y_bottom, concat_y_top);
		}
	}

	template <typename Dtype>
	void PRNNBaseLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top){
		//1. slice_h0_
		const int N = bottom[0]->shape(0);
		const vector<Blob<Dtype>*> slice_bottom(1, bottom[0]);
		vector<Blob<Dtype>*> slice_top(N, NULL);
		for (int n = 0; n < N; ++n){
			slice_top[n] = H0_[n].get();
		}
		slice_h0_->Forward(slice_bottom, slice_top);

		// 2. for all sequence, run LSTM
		for (int t = 0; t < T_; ++t){
			this->RecurrentForward(t);
			// 5. ip_h_
			const vector<Blob<Dtype>*> ip_h_bottom(1, H_[t].get());
			const vector<Blob<Dtype>*> ip_h_top(1, Y_[t].get());
			ip_h_->Forward(ip_h_bottom, ip_h_top);
		}

		// 6. concat_y_
		vector<Blob<Dtype>*> concat_y_bottom(T_, NULL);
		for (int t = 0; t < T_; ++t){
			concat_y_bottom[t] = Y_[t].get();
		}
		const vector<Blob<Dtype>*> concat_y_top(1, top[0]);
		concat_y_->Forward(concat_y_bottom, concat_y_top);
	}

	INSTANTIATE_CLASS(PRNNBaseLayer);
}