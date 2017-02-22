#include <string>
#include <utility>
#include <vector>


#include "caffe/filler.hpp"
#include "caffe/layers/drnn_base_layer.hpp"

namespace caffe{
	template <typename Dtype>
	void DRNNBaseLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top){
		// H0_: num_seq_, #streams, hidden_dim_
		CHECK_EQ(3, bottom[0]->num_axes());
		// cont_: (T_, #streams) 
		CHECK_EQ(2, bottom[1]->num_axes());
		CHECK_EQ(bottom[0]->shape(1), bottom[1]->shape(1));
		if (bottom[1]->shape(1) > 1){
			LOG(ERROR) << "Please make sure that each stream has the same 'cont' variable";
		}
		conditional_ = this->layer_param_.recurrent_param().conditional();
		LOG_IF(INFO, conditional_) << "Decode input is groundtruth input sequence";
		output_dim_ = this->layer_param_.recurrent_param().output_dim();
		vector<int> x_shape(3, 1);
		// if we need to delay the decoding input for 1 timestep
		// e.g. for decoding video sequence, we need a delay
		// for decoding sentence, we do not need a delay
		delay_ = this->layer_param_.recurrent_param().delay();
		if (!conditional_){
			CHECK(delay_) << "for un-conditonal decoding, delay must be set to true";
		}
		if (conditional_){
			CHECK_GE(bottom.size(), 3);
			//X_: T_, #streams, X_dim_
			CHECK_EQ(3, bottom[2]->num_axes());
			CHECK_EQ(bottom[1]->shape(0), bottom[2]->shape(0));
			CHECK_EQ(bottom[1]->shape(1), bottom[2]->shape(1));
			//shapes of blobs
			x_shape[1] = bottom[2]->shape(1);
			x_shape[2] = bottom[2]->shape(2);
			X_dim_ = bottom[2]->shape(2);
		}
		hidden_dim_ = bottom[0]->shape(2);
		T_ = bottom[1]->shape(0);
		num_seq_ = bottom[0]->shape(0);

		vector<int> h_shape(3, 1);
		h_shape[1] = bottom[0]->shape(1);
		h_shape[2] = bottom[0]->shape(2);
		vector<int> y_shape(3, 1);
		y_shape[1] = bottom[0]->shape(1);
		y_shape[2] = output_dim_;

		// setup slice_h_ layer
		// Top
		H0_.resize(num_seq_);
		for (int n = 0; n < num_seq_; ++n){
			H0_[n].reset(new Blob<Dtype>(h_shape));
		}
		// Layer
		const vector<Blob<Dtype>*> slice_h_bottom(1, bottom[0]);
		const vector<Blob<Dtype>*> slice_h_top(num_seq_, H0_[0].get());
		LayerParameter slice_param;
		slice_param.mutable_slice_param()->set_axis(0);
		slice_h_.reset(new SliceLayer<Dtype>(slice_param));
		slice_h_->SetUp(slice_h_bottom, slice_h_top);


		// setup slice_x_ layer
		// Top
		if (conditional_){
			X_.resize(T_);
			for (int t = 0; t < T_; ++t){
				X_[t].reset(new Blob<Dtype>(x_shape));
			}
			// Layer
			const vector<Blob<Dtype>*> slice_x_bottom(1, bottom[2]);
			const vector<Blob<Dtype>*> slice_x_top(T_, X_[0].get());
			slice_x_.reset(new SliceLayer<Dtype>(slice_param));
			slice_x_->SetUp(slice_x_bottom, slice_x_top);
		}
		
		// setup ip_h_ layer
		// Bottom && Top
		Y_.resize(T_);
		H_.resize(T_);
		for (int t = 0; t < T_; ++t){
			Y_[t].reset(new Blob<Dtype>(y_shape));
			H_[t].reset(new Blob<Dtype>(h_shape));
		}
		const vector<Blob<Dtype>*> ip_h_bottom(1, H_[0].get());
		const vector<Blob<Dtype>*> ip_h_top(1, Y_[0].get());
		// Layer
		LayerParameter ip_h_param(this->layer_param_);
		ip_h_param.mutable_inner_product_param()->set_num_output(output_dim_);
		ip_h_param.mutable_inner_product_param()->set_axis(2);
		ip_h_.reset(new InnerProductLayer<Dtype>(ip_h_param));
		ip_h_->SetUp(ip_h_bottom, ip_h_top);

		// if not conditional, setup split_y_ layer
		Y_1_.resize(T_);
		for (int t = 0; t < T_; ++t){
			Y_1_[t].reset(new Blob<Dtype>(y_shape));
		}
		if (!conditional_){
			// Top
			Y_2_.resize(T_);
			for (int t = 0; t < T_; ++t){
				Y_2_[t].reset(new Blob<Dtype>(y_shape));
			}
			const vector<Blob<Dtype>*> split_y_bottom(1, Y_[0].get());
			vector<Blob<Dtype>*> split_y_top(2, Y_1_[0].get());
			split_y_top[1] = Y_2_[0].get();
			// Layer
			split_y_.reset(new SplitLayer<Dtype>(LayerParameter()));
			split_y_->SetUp(split_y_bottom, split_y_top);
		}
		else{
			for (int t = 0; t < T_; ++t){
				Y_1_[t]->ShareData(*(Y_[t].get()));
				Y_1_[t]->ShareDiff(*(Y_[t].get()));
			}
		}

		// setup concat_y_ layer
		// Bottom && Top
		vector<Blob<Dtype>*> concat_y_bottom(T_, NULL);
		for (int t = 0; t < T_; ++t){
			concat_y_bottom[t] = Y_1_[t].get();
		}
		const vector<Blob<Dtype>*> concat_y_top(1, top[0]);
		// Layer
		LayerParameter concat_param;
		concat_param.mutable_concat_param()->set_axis(0);
		concat_y_.reset(new ConcatLayer<Dtype>(concat_param));
		concat_y_->SetUp(concat_y_bottom, concat_y_top);

		// setup zero_blob_ for beginning of the input
		// TODO: for unconditional decoding, 
		// allow to specify the zero_blob_ value manually
		if (conditional_){
			start_blob_.reset(new Blob<Dtype>(x_shape));
			zero_blob_.reset(new Blob<Dtype>(x_shape));
		}
		else{
			start_blob_.reset(new Blob<Dtype>(y_shape));
			zero_blob_.reset(new Blob<Dtype>(y_shape));
		}

		start_H_.reset(new Blob<Dtype>(h_shape));
	}

	template <typename Dtype>
	void DRNNBaseLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top){
		// check in every iteration, to deal with changed 
		// number of sequences in a batch
		// H0_: T_, #streams, hidden_dim_
		CHECK_EQ(3, bottom[0]->num_axes());
		CHECK_EQ(bottom[0]->shape(2), hidden_dim_)
			<< "H0_ feat dim incompatible with dlstm parameters.";
		// cont_: (T_, #streams) 
		CHECK_EQ(2, bottom[1]->num_axes());
		CHECK_EQ(bottom[0]->shape(1), bottom[1]->shape(1));
		vector<int> x_shape(3, 1);
		if (conditional_){
			//X_: T_, #streams, X_dim_
			CHECK_EQ(3, bottom[2]->num_axes());
			CHECK_EQ(bottom[1]->shape(0), bottom[2]->shape(0));
			CHECK_EQ(bottom[1]->shape(1), bottom[2]->shape(1));
			//shapes of blobs
			x_shape[1] = bottom[2]->shape(1);
			x_shape[2] = bottom[2]->shape(2);
			CHECK_EQ(bottom[2]->shape(2), X_dim_)
				<< "X feat dim incompatible with dlstm parameters.";
		}
		vector<int> h_shape(3, 1);
		h_shape[1] = bottom[0]->shape(1);
		h_shape[2] = bottom[0]->shape(2);
		vector<int> y_shape(3, 1);
		y_shape[1] = bottom[0]->shape(1);
		y_shape[2] = output_dim_;
		// because ip_g_ and ip_y_ is for element blob, their bottom and top
		// shape will not change, so we don't need to reshape them here
		if (bottom[0]->shape(0) != num_seq_){
			num_seq_ = bottom[0]->shape(0);
			H0_.resize(num_seq_);
			for (int n = 0; n < num_seq_; ++n){
				H0_[n].reset(new Blob<Dtype>(h_shape));
			}
			// reshape slice_h_
			const vector<Blob<Dtype>*> slice_h_bottom(1, bottom[0]);
			const vector<Blob<Dtype>*> slice_h_top(num_seq_, H0_[0].get());
			slice_h_->Reshape(slice_h_bottom, slice_h_top);
		}
		if (bottom[1]->shape(0) != T_){
			T_ = bottom[1]->shape(0);
			if (conditional_){
				X_.resize(T_);
				for (int t = 0; t < T_; ++t){
					X_[t].reset(new Blob<Dtype>(x_shape));
				}
				// reshape slice_x_
				const vector<Blob<Dtype>*> slice_x_bottom(1, bottom[2]);
				const vector<Blob<Dtype>*> slice_x_top(T_, X_[0].get());
				slice_x_->Reshape(slice_x_bottom, slice_x_top);
			}
			Y_.resize(T_);
			H_.resize(T_);
			for (int t = 0; t < T_; ++t){
				Y_[t].reset(new Blob<Dtype>(y_shape));
				H_[t].reset(new Blob<Dtype>(h_shape));
			}
			Y_1_.resize(T_);
			for (int t = 0; t < T_; ++t){
				Y_1_[t].reset(new Blob<Dtype>(y_shape));
			}
			if (!conditional_){
				Y_2_.resize(T_);
				for (int t = 0; t < T_; ++t){
					Y_2_[t].reset(new Blob<Dtype>(y_shape));
				}
			}
			else{
				for (int t = 0; t < T_; ++t){
					Y_1_[t]->ShareData(*(Y_[t].get()));
					Y_1_[t]->ShareDiff(*(Y_[t].get()));
				}
			}
			// reshape concat_y_
			vector<Blob<Dtype>*> concat_y_bottom(T_, NULL);
			for (int t = 0; t < T_; ++t){
				concat_y_bottom[t] = Y_1_[t].get();
			}
			const vector<Blob<Dtype>*> concat_y_top(1, top[0]);
			concat_y_->Reshape(concat_y_bottom, concat_y_top);
		}
		vector<int> top_shape(3, T_);
		top_shape[1] = bottom[0]->shape(1);
		top_shape[2] = output_dim_;
		top[0]->Reshape(top_shape);
	}

	template <typename Dtype>
	void DRNNBaseLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top){
		this->ShareWeight();
		// 1. slice_h_
		const vector<Blob<Dtype>*> slice_h_bottom(1, bottom[0]);
		vector<Blob<Dtype>*> slice_h_top(num_seq_, NULL);
		for (int n = 0; n < num_seq_; ++n){
			slice_h_top[n] = H0_[n].get();
		}
		slice_h_->Forward(slice_h_bottom, slice_h_top);

		// 3. slice_x_ if needed
		if (conditional_){
			const vector<Blob<Dtype>*> slice_x_bottom(1, bottom[2]);
			vector<Blob<Dtype>*> slice_x_top(T_, NULL);
			for (int t = 0; t < T_; ++t){
				slice_x_top[t] = X_[t].get();
			}
			slice_x_->Forward(slice_x_bottom, slice_x_top);
		}

		// for all sequence, run decode lstm.
		int seq_id = -1;
		int cont_t;
		const Dtype* cont_data = bottom[1]->cpu_data();
		const int cont_dim = bottom[1]->count(1);
		for (int t = 0; t < T_; t++){
			// NOTE: only take the cont of first stream as reference
			// maybe a bug here
			cont_t = static_cast<int>(cont_data[t * cont_dim]);
			if (cont_t == 0){
				seq_id++;
			}
			this->RecurrentForward(t, cont_t, seq_id);
			// 9. ip_h_
			const vector<Blob<Dtype>*> ip_h_bottom(1, H_[t].get());
			const vector<Blob<Dtype>*> ip_h_top(1, Y_[t].get());
			ip_h_->Forward(ip_h_bottom, ip_h_top);
			// 10. split_y_ if needed
			if (!conditional_){
				const vector<Blob<Dtype>*> split_y_bottom(1, Y_[t].get());
				vector<Blob<Dtype>*> split_y_top(2, Y_1_[t].get());
				split_y_top[1] = Y_2_[t].get();
				split_y_->Forward(split_y_bottom, split_y_top);
			}
		}

		// just data
		this->CopyRecurrentOutputAndInput();
		
		// 11. concat Y_1_
		vector<Blob<Dtype>*> concat_y_bottom(T_, NULL);
		for (int t = 0; t < T_; ++t){
			concat_y_bottom[t] = Y_1_[t].get();
		}
		const vector<Blob<Dtype>*> concat_y_top(1, top[0]);
		concat_y_->Forward(concat_y_bottom, concat_y_top);
	}

	template <typename Dtype>
	void DRNNBaseLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom){
		this->ShareWeight();

		// 11. concat Y_1_
		vector<Blob<Dtype>*> concat_y_bottom(T_, NULL);
		for (int t = 0; t < T_; ++t){
			concat_y_bottom[t] = Y_1_[t].get();
		}
		const vector<Blob<Dtype>*> concat_y_top(1, top[0]);
		concat_y_->Backward(concat_y_top,
			vector<bool>(T_, true),
			concat_y_bottom);

		// for all sequence, run decode LSTM
		int seq_id = num_seq_;
		int cont_t;
		const int cont_dim = bottom[1]->count(1);
		const Dtype* cont_data = bottom[1]->cpu_data();
		for (int t = T_ - 1; t >= 0; --t){
			// NOTE: only take the cont of first stream as reference
			// maybe a bug here
			cont_t = static_cast<int>(cont_data[t * cont_dim]);
			if (cont_t == 0){
				seq_id--;
			}
			// 10. split_y_ if needed
			if (!conditional_){
				const vector<Blob<Dtype>*> split_y_bottom(1, Y_[t].get());
				vector<Blob<Dtype>*> split_y_top(2, Y_1_[t].get());
				split_y_top[1] = Y_2_[t].get();
				split_y_->Backward(
					split_y_top,
					vector<bool>(1, true),
					split_y_bottom);
			}
			// 9. ip_h_
			const vector<Blob<Dtype>*> ip_h_bottom(1, H_[t].get());
			const vector<Blob<Dtype>*> ip_h_top(1, Y_[t].get());
			ip_h_->Backward(
				ip_h_top, 
				vector<bool>(1, true),
				ip_h_bottom);
			this->RecurrentBackward(t, cont_t, seq_id);
		}

		// 3. slice_x_ if needed
		if (conditional_){
			const vector<Blob<Dtype>*> slice_x_bottom(1, bottom[2]);
			vector<Blob<Dtype>*> slice_x_top(T_, NULL);
			for (int t = 0; t < T_; ++t){
				slice_x_top[t] = X_[t].get();
			}
			slice_x_->Backward(slice_x_top,
				vector<bool>(T_, true),
				slice_x_bottom);
		}

		
		// 1. slice_h_
		const vector<Blob<Dtype>*> slice_h_bottom(1, bottom[0]);
		vector<Blob<Dtype>*> slice_h_top(num_seq_, NULL);
		for (int n = 0; n < num_seq_; ++n){
			slice_h_top[n] = H0_[n].get();
		}
		slice_h_->Backward(slice_h_top,
			vector<bool>(num_seq_, true),
			slice_h_bottom);
	}

	INSTANTIATE_CLASS(DRNNBaseLayer);
} // namespace caffe
