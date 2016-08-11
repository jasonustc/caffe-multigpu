#include <string>
#include <utility>
#include <vector>

#include "caffe/layers/rnn_base_layer.hpp"

namespace caffe {

	template <typename Dtype>
	void RNNBaseLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top)
	{
		CHECK_EQ(3, bottom[0]->num_axes());
		CHECK_EQ(2, bottom[1]->num_axes());
		CHECK_EQ(bottom[0]->shape(0), bottom[1]->shape(0));
		CHECK_EQ(bottom[0]->shape(1), bottom[1]->shape(1));

		this->hidden_dim_ = this->GetHiddenDim();
		T_ = bottom[0]->shape(0);
		X_dim_ = bottom[0]->shape(2);

		// shapes of blobs
		/*
		const vector<int> x_shape {
			1, 
			bottom[0]->shape(1),
			bottom[0]->shape(2)
		};*/
		vector<int> x_shape(3, 1);
		x_shape[1] = bottom[0]->shape(1);
		x_shape[2] = bottom[0]->shape(2);
		/*
		const vector<int> h_shape{
			1,
			bottom[0]->shape(1),
			hidden_dim_
		};*/
		vector<int> h_shape(3, 1);
		h_shape[1] = bottom[0]->shape(1);
		h_shape[2] = hidden_dim_;
		/*
		const vector<int> cont_shape{
			1,
			bottom[0]->shape(1)
		};*/
		vector<int> cont_shape(2, 1);
		cont_shape[1] = bottom[0]->shape(1);

		// setup slice_x_ layer
		// Top
		X_.resize(T_);
		for (int t = 0; t < T_; ++t)
		{
			X_[t].reset(new Blob<Dtype>(x_shape));
		}
		// Layer
		const vector<Blob<Dtype>*> slice_x_bottom(1, bottom[0]);
		const vector<Blob<Dtype>*> slice_x_top(T_, X_[0].get());
		LayerParameter slice_param;
		slice_param.mutable_slice_param()->set_axis(0);
		slice_x_.reset(new SliceLayer<Dtype>(slice_param));
		slice_x_->SetUp(slice_x_bottom, slice_x_top);

		// setup slice_cont_ layer
		// Top
		CONT_.resize(T_);
		for (int t = 0; t < T_; ++t)
		{
			CONT_[t].reset(new Blob<Dtype>(cont_shape));
		}
		// Layer
		const vector<Blob<Dtype>*> slice_cont_bottom(1, bottom[1]);
		const vector<Blob<Dtype>*> slice_cont_top(T_, CONT_[0].get());
		slice_cont_.reset(new SliceLayer<Dtype>(slice_param));
		slice_cont_->SetUp(slice_cont_bottom, slice_cont_top);

		// setup concat_top_ layer
		// Bottom
		H_.resize(T_);
		for (int t = 0; t < T_; ++t)
		{
			H_[t].reset(new Blob<Dtype>(h_shape));
		}
		// Layer
		const vector<Blob<Dtype>*> concat_ht_bottom(T_, H_[0].get());
		const vector<Blob<Dtype>*> concat_ht_top(1, top[0]);
		LayerParameter concat_ht_param;
		concat_ht_param.mutable_concat_param()->set_axis(0);
		concat_ht_.reset(new ConcatLayer<Dtype>(concat_ht_param));
		concat_ht_->SetUp(concat_ht_bottom, concat_ht_top);
	}

	template <typename Dtype>
	void RNNBaseLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		// generally, we assume that hidden_dim_ will not change across
		// all the training samples
		CHECK_EQ(3, bottom[0]->num_axes());
		CHECK_EQ(2, bottom[1]->num_axes());
		CHECK_EQ(bottom[0]->shape(0), bottom[1]->shape(0));
		CHECK_EQ(bottom[0]->shape(1), bottom[1]->shape(1));
		CHECK_EQ(bottom[0]->shape(2), X_dim_)
			<< "input feat dim do not compatible with lstm paramters";
		if (bottom[0]->shape(0) != T_){
			T_ = bottom[0]->shape(0);

			// shapes of blobs
			/*
			const vector<int> x_shape{
				1,
				bottom[0]->shape(1),
				bottom[0]->shape(2)
			};*/
			vector<int> x_shape(3, 1);
			x_shape[1] = bottom[0]->shape(1);
			x_shape[2] = bottom[0]->shape(2);
			vector<int> h_shape(3, 1);
			h_shape[1] = bottom[0]->shape(1);
			h_shape[2] = this->hidden_dim_;
			/*
			const vector<int> cont_shape{
				1,
				bottom[0]->shape(1)
			};*/
			vector<int> cont_shape(2, 1);
			cont_shape[1] = bottom[0]->shape(1);

			// reshape slice_x_
			X_.resize(T_);
			for (int t = 0; t < T_; ++t){
				X_[t].reset(new Blob<Dtype>(x_shape));
			}
			const vector<Blob<Dtype>*> slice_x_bottom(1, bottom[0]);
			const vector<Blob<Dtype>*> slice_x_top(T_, X_[0].get());
			slice_x_->Reshape(slice_x_bottom, slice_x_top);

			// reshape slice_cont_
			CONT_.resize(T_);
			for (int t = 0; t < T_; ++t){
				CONT_[t].reset(new Blob<Dtype>(cont_shape));
			}
			const vector<Blob<Dtype>*> slice_cont_bottom(1, bottom[1]);
			const vector<Blob<Dtype>*> slice_cont_top(T_, CONT_[0].get());
			slice_cont_->Reshape(slice_cont_bottom, slice_cont_top);

			// reshape concat_ht_
			H_.resize(T_);
			for (int t = 0; t < T_; ++t){
				H_[t].reset(new Blob<Dtype>(h_shape));
			}
			const vector<Blob<Dtype>*> concat_ht_bottom(T_, H_[0].get());
			const vector<Blob<Dtype>*> concat_ht_top(1, top[0]);
			concat_ht_->Reshape(concat_ht_bottom, concat_ht_top);
		}
		vector<int> top_shape = bottom[0]->shape();
		top_shape[2] = this->hidden_dim_;
		top[0]->Reshape(top_shape);
	}

	template <typename Dtype>
	void RNNBaseLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		// every time refresh the address of data blobs to deal with
		// changed length of T_ or batch_size

		this->ShareWeight();

		// 1. slice_x_
		const vector<Blob<Dtype>*> slice_x_bottom(1, bottom[0]);
		vector<Blob<Dtype>*> slice_x_top(T_, NULL);
		for (int t = 0; t < T_; ++t)
		{
			slice_x_top[t] = X_[t].get();
		}
		slice_x_->Forward(slice_x_bottom, slice_x_top);

		// 2. slice_cont_
		const vector<Blob<Dtype>*> slice_cont_bottom(1, bottom[1]);
		vector<Blob<Dtype>*> slice_cont_top(T_, NULL);
		for (int t = 0; t < T_; ++t)
		{
			slice_cont_top[t] = CONT_[t].get();
		}
		slice_cont_->Forward(slice_cont_bottom, slice_cont_top);

		// For all sequence run lstm.
		for (int t = 0; t < T_; t++)
		{
			this->RecurrentForward(t);
		}
		this->CopyRecurrentOutput();

		//9. concat_ht_
		vector<Blob<Dtype>*> concat_ht_bottom(T_, NULL);
		for (int t = 0; t < T_; ++t)
		{
			concat_ht_bottom[t] = H_[t].get();
		}
		const vector<Blob<Dtype>*> concat_ht_top(1, top[0]);
		concat_ht_->Forward(concat_ht_bottom, concat_ht_top);
	}

	template <typename Dtype>
	void RNNBaseLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

		this->ShareWeight();

		//9. concat_ht_
		vector<Blob<Dtype>*> concat_ht_bottom(T_, NULL);
		for (int t = 0; t < T_; ++t)
		{
			concat_ht_bottom[t] = H_[t].get();
		}
		const vector<Blob<Dtype>*> concat_ht_top(1, top[0]);
		concat_ht_->Backward(concat_ht_top,
			vector<bool>(T_, true),
			concat_ht_bottom);

		// For all sequence run lstm.
		for (int t = T_ - 1; t >= 0; t--)
		{
			this->RecurrentBackward(t);
		}

		// 1. slice_x_
		const vector<Blob<Dtype>*> slice_x_bottom(1, bottom[0]);
		vector<Blob<Dtype>*> slice_x_top(T_, NULL);
		for (int t = 0; t < T_; ++t)
		{
			slice_x_top[t] = X_[t].get();
		}
		slice_x_->Backward(slice_x_top,
			vector<bool>(1, true),
			slice_x_bottom);
	}

	INSTANTIATE_CLASS(RNNBaseLayer);
}  // namespace caffe
