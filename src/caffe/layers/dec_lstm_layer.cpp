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
		// parameters and layers
		bias_term_ = this->layer_param_.inner_product_param().bias_term();
		if (!bias_term_){
			this->blobs_.resize(2);
		}
		else{
			this->blobs_.resize(4);
		}

		//shapes of blobs
		int x_dim = this->conditional_ ? bottom[3]->shape(2) : this->output_dim_;
    /*
		const vector<int> x_shape{
			1,
			bottom[0]->shape(1),
			x_dim
		};*/
		vector<int> x_shape(3, 1);
		x_shape[1] = bottom[0]->shape(1);
		x_shape[2] = x_dim;
    /*
		const vector<int> h_shape{
			1,
			bottom[0]->shape(1),
			this->hidden_dim_
		};*/
		vector<int> h_shape(3, 1);
		h_shape[1] = bottom[0]->shape(1);
		h_shape[2] = this->hidden_dim_;
    /*
		const vector<int> xh_shape{
			1,
			bottom[0]->shape(1),
			x_dim + hidden_dim_
		};*/
		vector<int> xh_shape(3, 1);
		xh_shape[1] = bottom[0]->shape(1);
		xh_shape[2] = x_dim + this->hidden_dim_;
    /*
		const vector<int> gate_shape{
			1,
			bottom[0]->shape(1),
			hidden_dim_ * 4
		};*/
		vector<int> gate_shape(3, 1);
		gate_shape[1] = bottom[0]->shape(1);
		gate_shape[2] = this->hidden_dim_ * 4;

		// setup split_h_ layer
		H_1_.resize(this->T_);
		H_2_.resize(this->T_);
		for (int t = 0; t < this->T_; ++t){
			H_1_[t].reset(new Blob<Dtype>(h_shape));
			H_2_[t].reset(new Blob<Dtype>(h_shape));
		}
		const vector<Blob<Dtype>*> split_h_bottom(1, this->H_[0].get());
    /*
		const vector<Blob<Dtype>*> split_h_top{
			H_1_[0].get(),
			H_2_[0].get()
		};*/
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
    /*
		const vector<Blob<Dtype>*> concat_bottom{
			this->X_[0].get(),
			this->H_[0].get()
		};*/
		vector<Blob<Dtype>*> concat_bottom(2, NULL);
		concat_bottom[0] = this->X_[0].get();
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
    /*
		const vector<Blob<Dtype>*> dlstm_unit_bottom{
			C_[0].get(),
			G_[0].get()
		};*/
		vector<Blob<Dtype>*> dlstm_unit_bottom(2, C_[0].get());
		dlstm_unit_bottom[1] = G_[0].get();
    /*
		const vector<Blob<Dtype>*> dlstm_unit_top{
			C_[0].get(),
			this->H_[0].get()
		};*/
		vector<Blob<Dtype>*> dlstm_unit_top(2, C_[0].get());
		dlstm_unit_top[1] = this->H_[0].get();
		//Layer
		dlstm_unit_.reset(new DLSTMUnitLayer<Dtype>(LayerParameter()));
		dlstm_unit_->SetUp(dlstm_unit_bottom, dlstm_unit_top);
	}

	template <typename Dtype>
	void DLSTMLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top){
		DRNNBaseLayer<Dtype>::Reshape(bottom, top);
		//TODO: reshape XH_ and G_
		// length of sequence has changed
		if (C_.size() != this->H_.size()){
      /*
			const vector<int> h_shape{
			1,
			bottom[0]->shape(1),
			this->hidden_dim_
			};*/
			vector<int> h_shape(3, 1);
			h_shape[1] = bottom[0]->shape(1);
			h_shape[2] = this->hidden_dim_;

			vector<int> xh_shape(3, 1);
			xh_shape[1] = bottom[0]->shape(1);
			xh_shape[2] = this->hidden_dim_ + this->output_dim_;

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
			/// concat_bottom[0] = start_blob_.get();
			concat_bottom[0] = this->delay_ ? this->start_blob_.get() : this->X_[t].get();
			concat_bottom[1] = this->H0_[seq_id].get();
		}
		else{
			if (this->conditional_){
				/// concat_bottom[0] = X_[t - 1].get();
				concat_bottom[0] = this->delay_ ? this->X_[t - 1].get() : this->X_[t].get();
			}
			else{
				concat_bottom[0] = this->Y_2_[t - 1].get();
			}
			concat_bottom[1] = H_2_[t - 1].get();
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
			dlstm_unit_bottom[0] = C_[t - 1].get();
		}
		dlstm_unit_bottom[1] = G_[t].get();
    /*
		const vector<Blob<Dtype>*> dlstm_unit_top{
			C_[t].get(),
			H_1_[t].get()
		};*/
		vector<Blob<Dtype>*> dlstm_unit_top(2, C_[t].get());
		dlstm_unit_top[1] = H_1_[t].get();
		dlstm_unit_->Forward(dlstm_unit_bottom, dlstm_unit_top);

		// 7. split
		const vector<Blob<Dtype>*> split_h_bottom(1, H_1_[t].get());
		/*const vector<Blob<Dtype>*> split_h_top{ this->H_[t].get(), H_2_[t].get() };*/
		vector<Blob<Dtype>*> split_h_top(2, this->H_[t].get());
    split_h_top[1] = H_2_[t].get();
		split_h_->Forward(split_h_bottom, split_h_top);
	}

	template <typename Dtype>
	void DLSTMLayer<Dtype>::RecurrentBackward(const int t, const int cont_t,
		const int seq_id){
		// 7. split
		const vector<Blob<Dtype>*> split_h_bottom(1, H_1_[t].get());
		/*const vector<Blob<Dtype>*> split_h_top{ this->H_[t].get(), H_2_[t].get() };*/
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
			dlstm_unit_bottom[0] = C_[t - 1].get();
		}
		dlstm_unit_bottom[1] = G_[t].get();
    /*
		const vector<Blob<Dtype>*> dlstm_unit_top{
			C_[t].get(),
			H_1_[t].get()
		};*/
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
			/// concat_bottom[0] = start_blob_.get();
			concat_bottom[0] = this->delay_ ? this->start_blob_.get() : this->X_[t].get();
			concat_bottom[1] = this->H0_[seq_id].get();
		}
		else{
			if (this->conditional_){
				/// concat_bottom[0] = X_[t - 1].get();
				concat_bottom[0] = this->delay_ ? this->X_[t - 1].get() : this->X_[t].get();
			}
			else{
				concat_bottom[0] = this->Y_2_[t - 1].get();
			}
			concat_bottom[1] = H_2_[t - 1].get();
		}
		vector<Blob<Dtype>*> concat_top(1, XH_[t].get());
		concat_->Backward(
			concat_top, 
			vector<bool>(2, true),
			concat_bottom);
	}

#ifdef CPU_ONLY
	STUB_GPU(DLSTMLayer);
#endif

	INSTANTIATE_CLASS(DLSTMLayer);
	REGISTER_LAYER_CLASS(DLSTM);
} // namespace caffe
