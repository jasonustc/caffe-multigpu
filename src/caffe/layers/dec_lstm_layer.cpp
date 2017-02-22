#include <utility>
#include <vector>

#include "caffe/layers/dec_lstm_layer.hpp"

namespace caffe{
	/*
	 * bottom[0]->shape(1): #streams
	 */
	template <typename Dtype>
	void DLSTMLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top){
		DRNNBaseLayer<Dtype>::LayerSetUp(bottom, top);
		has_c0_ = this->layer_param_.recurrent_param().has_c0_id();
		if (has_c0_){
			c0_id_ = this->layer_param_.recurrent_param().c0_id();
			LOG(INFO) << "using external c0: bottom[" << c0_id_ << "]";
			CHECK_LT(c0_id_, bottom.size());
			// C0_: num_seq_, #streams, hidden_dim_
			CHECK_EQ(3, bottom[c0_id_]->num_axes());
			CHECK(bottom[0]->shape() == bottom[c0_id_]->shape());
		}
		// parameters and layers
		bias_term_ = this->layer_param_.inner_product_param().bias_term();
		if (!bias_term_){
			this->blobs_.resize(2);
		}
		else{
			this->blobs_.resize(4);
		}

		//shapes of blobs
		int x_dim = this->conditional_ ? bottom[2]->shape(2) : this->output_dim_;
		vector<int> x_shape(3, 1);
		x_shape[1] = bottom[0]->shape(1);
		x_shape[2] = x_dim;
		vector<int> h_shape(3, 1);
		h_shape[1] = bottom[0]->shape(1);
		h_shape[2] = this->hidden_dim_;
		vector<int> xh_shape(3, 1);
		xh_shape[1] = bottom[0]->shape(1);
		xh_shape[2] = x_dim + this->hidden_dim_;
		vector<int> gate_shape(3, 1);
		gate_shape[1] = bottom[0]->shape(1);
		gate_shape[2] = this->hidden_dim_ * 4;

		// setup slice_c_ layer
		// Top
		C0_.resize(this->num_seq_);
		for (int n = 0; n < this->num_seq_; ++n){
			C0_[n].reset(new Blob<Dtype>(h_shape));
		}

		// Layer
		// if no external c0s, we just use zero c0s
		if (has_c0_){
			LayerParameter slice_param;
			slice_param.mutable_slice_param()->set_axis(0);
			const vector<Blob<Dtype>*> slice_c_bottom(1, bottom[c0_id_]);
			const vector<Blob<Dtype>*> slice_c_top(this->num_seq_, C0_[0].get());
			slice_c_.reset(new SliceLayer<Dtype>(slice_param));
			slice_c_->SetUp(slice_c_bottom, slice_c_top);
		}

		// setup split_h_ layer
		H_1_.resize(this->T_);
		H_2_.resize(this->T_);
		for (int t = 0; t < this->T_; ++t){
			H_1_[t].reset(new Blob<Dtype>(h_shape));
			H_2_[t].reset(new Blob<Dtype>(h_shape));
		}
		const vector<Blob<Dtype>*> split_h_bottom(1, this->H_[0].get());
		vector<Blob<Dtype>*> split_h_top(2, H_1_[0].get());
		split_h_top[1] = H_2_[0].get();
		split_h_.reset(new SplitLayer<Dtype>(LayerParameter()));
		split_h_->SetUp(split_h_bottom, split_h_top);

		// setup concat_ layer
		// Bottom && Top
		XH_.resize(this->T_);
		for (int t = 0; t < this->T_; ++t){
			XH_[t].reset(new Blob<Dtype>(xh_shape));
		}
		// Layer
		vector<Blob<Dtype>*> concat_bottom(2, NULL);
		if (this->conditional_){
			concat_bottom[0] = this->X_[0].get();
		}
		else{
			concat_bottom[0] = this->Y_2_[0].get();
		}
		concat_bottom[1] = this->H_[0].get();
		const vector<Blob<Dtype>*> concat_top(1, XH_[0].get());
		LayerParameter concat_param;
		concat_param.mutable_concat_param()->set_axis(2);
		concat_.reset(new ConcatLayer<Dtype>(concat_param));
		concat_->SetUp(concat_bottom, concat_top);

		// setup ip_g_ layer
		// Top
		G_.resize(this->T_);
		for (int t = 0; t < this->T_; ++t){
			G_[t].reset(new Blob<Dtype>(gate_shape));
		}
		const vector<Blob<Dtype>*> ip_g_bottom(1, XH_[0].get());
		const vector<Blob<Dtype>*> ip_g_top(1, G_[0].get());
		//Layer
		LayerParameter ip_g_param(this->layer_param_);
		ip_g_param.mutable_inner_product_param()->set_num_output(4 * this->hidden_dim_);
		ip_g_param.mutable_inner_product_param()->set_axis(2);
		ip_g_.reset(new InnerProductLayer<Dtype>(ip_g_param));
		ip_g_->SetUp(ip_g_bottom, ip_g_top);

		this->blobs_[0].reset(new Blob<Dtype>(this->ip_h_->blobs()[0]->shape()));
		this->blobs_[0]->ShareData(*(this->ip_h_->blobs())[0]);
		this->blobs_[0]->ShareData(*(this->ip_h_->blobs())[0]);

		this->blobs_[1].reset(new Blob<Dtype>(ip_g_->blobs()[0]->shape()));
		this->blobs_[1]->ShareData(*(ip_g_->blobs())[0]);
		this->blobs_[1]->ShareDiff(*(ip_g_->blobs())[0]);

		if (bias_term_){
			this->blobs_[2].reset(new Blob<Dtype>(this->ip_h_->blobs()[1]->shape()));
			this->blobs_[2]->ShareData(*(this->ip_h_->blobs())[1]);
			this->blobs_[2]->ShareData(*(this->ip_h_->blobs())[1]);
			this->blobs_[3].reset(new Blob<Dtype>(ip_g_->blobs()[1]->shape()));
			this->blobs_[3]->ShareData(*(ip_g_->blobs())[1]);
			this->blobs_[3]->ShareDiff(*(ip_g_->blobs())[1]);
		}

		// setup dlstm_unit_ layer
		// Bottom
		C_.resize(this->T_);
		for (int t = 0; t < this->T_; ++t){
			C_[t].reset(new Blob<Dtype>(h_shape));
		}
		vector<Blob<Dtype>*> dlstm_unit_bottom(2, C_[0].get());
		dlstm_unit_bottom[1] = G_[0].get();
		vector<Blob<Dtype>*> dlstm_unit_top(2, C_[0].get());
		dlstm_unit_top[1] = this->H_[0].get();
		//Layer
		dlstm_unit_.reset(new DLSTMUnitLayer<Dtype>(LayerParameter()));
		dlstm_unit_->SetUp(dlstm_unit_bottom, dlstm_unit_top);

		// start_C_
		start_C_.reset(new Blob<Dtype>(h_shape));
	}

	template <typename Dtype>
	void DLSTMLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top){
		vector<int> h_shape(3, 1);
		h_shape[1] = bottom[0]->shape(1);
		h_shape[2] = bottom[0]->shape(2);
		DRNNBaseLayer<Dtype>::Reshape(bottom, top);
		// C0_: T_, #streams, hidden_dim_
		if (has_c0_){
			CHECK_EQ(3, bottom[c0_id_]->num_axes());
			CHECK_EQ(bottom[c0_id_]->shape(2), this->hidden_dim_)
				<< "C0_ feat dim incompatible with dlstm parameters.";
			CHECK(bottom[0]->shape() == bottom[c0_id_]->shape()) << bottom[0]->shape_string()
				<< " vs. " << bottom[c0_id_]->shape_string();
		}
		if (C0_.size() != this->num_seq_){
			C0_.resize(this->num_seq_);
			for (int n = 0; n < this->num_seq_; ++n){
				C0_[n].reset(new Blob<Dtype>(h_shape));
			}
			if (has_c0_){
				// reshape slice_c_
				const vector<Blob<Dtype>*> slice_c_bottom(1, bottom[c0_id_]);
				const vector<Blob<Dtype>*> slice_c_top(this->num_seq_, C0_[0].get());
				slice_c_->Reshape(slice_c_bottom, slice_c_top);
			}
		}

		// length of sequence has changed
		if (C_.size() != this->H_.size()){
			vector<int> h_shape(3, 1);
			h_shape[1] = bottom[0]->shape(1);
			h_shape[2] = this->hidden_dim_;

			vector<int> xh_shape(3, 1);
			int x_dim = this->conditional_ ? bottom[2]->shape(2) : this->output_dim_;
			xh_shape[1] = bottom[0]->shape(1);
			xh_shape[2] = this->hidden_dim_ + x_dim;

			vector<int> gate_shape(3, 1);
			gate_shape[1] = bottom[0]->shape(1);
			gate_shape[2] = this->hidden_dim_ * 4;

			XH_.resize(this->T_);
			G_.resize(this->T_);
			C_.resize(this->T_);
			H_1_.resize(this->T_);
			H_2_.resize(this->T_);
			for (int t = 0; t < this->T_; ++t){
				XH_[t].reset(new Blob<Dtype>(xh_shape));
				G_[t].reset(new Blob<Dtype>(gate_shape));
				C_[t].reset(new Blob<Dtype>(h_shape));
				H_1_[t].reset(new Blob<Dtype>(h_shape));
				H_2_[t].reset(new Blob<Dtype>(h_shape));
			}
		}
	}

	template <typename Dtype>
	void DLSTMLayer<Dtype>::RecurrentForward(const int t, const int cont_t,
		const int seq_id){
		// 4. concat input_t and h_{t - 1}
		vector<Blob<Dtype>*> concat_bottom(2, NULL);
		if (!cont_t){
			// begin of a sequence
			concat_bottom[0] = this->delay_ ? this->zero_blob_.get(): this->X_[t].get();
			concat_bottom[1] = this->H0_[seq_id].get();
		}
		else{
			if (this->conditional_){
				concat_bottom[0] = this->delay_ ? 
					(t == 0 ? this->start_blob_.get() : this->X_[t - 1].get()) 
					: this->X_[t].get();
			}
			else{
				// in case that the start of this batch is not the start of a sequence
				concat_bottom[0] = t == 0 ? this->start_blob_.get() : this->Y_2_[t - 1].get();
			}
			// in case that the start of this batch is not the start of a sequence
			concat_bottom[1] = t == 0 ? this->start_H_.get(): H_2_[t - 1].get();
		}
		vector<Blob<Dtype>*> concat_top(1, XH_[t].get());
		concat_->Forward(concat_bottom, concat_top);

		// 5. forward gate
		const vector<Blob<Dtype>*> ip_g_bottom(1, XH_[t].get());
		const vector<Blob<Dtype>*> ip_g_top(1, G_[t].get());
		ip_g_->Forward(ip_g_bottom, ip_g_top);

		// 6. dlstm unit
		vector<Blob<Dtype>*> dlstm_unit_bottom(2, NULL);
		if (!cont_t){
			//begin of a sequence
			dlstm_unit_bottom[0] = this->C0_[seq_id].get();
		}
		else{
			// in case that the start of this batch is not the start of a sequence
			dlstm_unit_bottom[0] = t == 0 ? start_C_.get() : C_[t - 1].get();
		}
		dlstm_unit_bottom[1] = G_[t].get();
		vector<Blob<Dtype>*> dlstm_unit_top(2, C_[t].get());
		dlstm_unit_top[1] = H_1_[t].get();
		dlstm_unit_->Forward(dlstm_unit_bottom, dlstm_unit_top);

		// 7. split
		const vector<Blob<Dtype>*> split_h_bottom(1, H_1_[t].get());
		vector<Blob<Dtype>*> split_h_top(2, this->H_[t].get());
		split_h_top[1] = H_2_[t].get();
		split_h_->Forward(split_h_bottom, split_h_top);
	}

	template <typename Dtype>
	void DLSTMLayer<Dtype>::RecurrentBackward(const int t, const int cont_t,
		const int seq_id){
		// 7. split
		const vector<Blob<Dtype>*> split_h_bottom(1, H_1_[t].get());
		vector<Blob<Dtype>*> split_h_top(2, this->H_[t].get()); 
		split_h_top[1] = H_2_[t].get();
		split_h_->Backward(split_h_top,
			vector<bool>(1, true),
			split_h_bottom);

		// 6. dlstm unit
		vector<Blob<Dtype>*> dlstm_unit_bottom(2, NULL);
		if (!cont_t){
			//begin of a sequence
			dlstm_unit_bottom[0] = this->C0_[seq_id].get();
		}
		else{
			// in case that the start of this batch is not the start of a sequence
			dlstm_unit_bottom[0] = t == 0 ? start_C_.get() : C_[t - 1].get();
		}
		dlstm_unit_bottom[1] = G_[t].get();
		vector<Blob<Dtype>*> dlstm_unit_top(2, C_[t].get());
		dlstm_unit_top[1] = H_1_[t].get();
		dlstm_unit_->Backward(
			dlstm_unit_top,
			vector<bool>(2, true),
			dlstm_unit_bottom);

		// 5.backward gate
		const vector<Blob<Dtype>*> ip_g_bottom(1, XH_[t].get());
		const vector<Blob<Dtype>*> ip_g_top(1, G_[t].get());
		ip_g_->Backward(ip_g_top,
			vector<bool>(1, true),
			ip_g_bottom);

		// 4. concat
		vector<Blob<Dtype>*> concat_bottom(2, NULL);
		if (!cont_t){
			// begin of a sequence
			concat_bottom[0] = this->delay_ ? this->zero_blob_.get() : this->X_[t].get();
			concat_bottom[1] = this->H0_[seq_id].get();
		}
		else{
			if (this->conditional_){
				// in case that the start of this batch is not the start of a sequence
				concat_bottom[0] = this->delay_ ? 
					(t == 0 ? this->start_blob_.get() : this->X_[t - 1].get())
					: this->X_[t].get();
			}
			else{
				// in case that the start of this batch is not the start of a sequence
				concat_bottom[0] = t == 0 ? this->start_blob_.get() : this->Y_2_[t - 1].get();
			}
			// in case that the start of this batch is not the start of a sequence
			concat_bottom[1] = t == 0 ? this->start_H_.get() : H_2_[t - 1].get();
		}
		vector<Blob<Dtype>*> concat_top(1, XH_[t].get());
		concat_->Backward(
			concat_top, 
			vector<bool>(2, true),
			concat_bottom);
	}

	template <typename Dtype>
	void DLSTMLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top){
		// 2. slice_c_ 
		if (has_c0_){
			const vector<Blob<Dtype>*> slice_c_bottom(1, bottom[c0_id_]);
			vector<Blob<Dtype>*> slice_c_top(this->num_seq_, NULL);
			for (int n = 0; n < this->num_seq_; ++n){
				slice_c_top[n] = C0_[n].get();
			}
			slice_c_->Forward(slice_c_bottom, slice_c_top);
		}

		DRNNBaseLayer<Dtype>::Forward_cpu(bottom, top);
	}

	template<typename Dtype>
	void DLSTMLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down,
		const vector<Blob<Dtype>*>& bottom){
		DRNNBaseLayer<Dtype>::Backward_cpu(top, propagate_down, bottom);
		// 2. slice_c_
		if (has_c0_){
			const vector<Blob<Dtype>*> slice_c_bottom(1, bottom[c0_id_]);
			vector<Blob<Dtype>*> slice_c_top(this->num_seq_, NULL);
			for (int n = 0; n < this->num_seq_; ++n){
				slice_c_top[n] = C0_[n].get();
			}
			slice_c_->Backward(slice_c_top,
				vector<bool>(this->num_seq_, true),
				slice_c_bottom);
		}
	}

#ifdef CPU_ONLY
	STUB_GPU(DLSTMLayer);
#endif

	INSTANTIATE_CLASS(DLSTMLayer);
	REGISTER_LAYER_CLASS(DLSTM);
} // namespace caffe
