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
			blobs_.resize(2);
		}
		else{
			blobs_.resize(4);
		}

		//shapes of blobs
		int x_dim = conditional_ ? bottom[3]->shape(2) : output_dim_;
		const vector<int> x_shape{
			1,
			bottom[0]->shape(1),
			x_dim
		};
		const vector<int> h_shape{
			1,
			bottom[0]->shape(1),
			hidden_dim_
		};
		const vector<int> xh_shape{
			1,
			bottom[0]->shape(1),
			x_dim + hidden_dim_
		};
		const vector<int> gate_shape{
			1,
			bottom[0]->shape(1),
			hidden_dim_ * 4
		};

		// setup concat_ layer
		// Bottom && Top
		XH_.resize(T_);
		for (int t = 0; t < T_; ++t){
			XH_[t].reset(new Blob<Dtype>(xh_shape));
		}
		// Layer
		const vector<Blob<Dtype>*> concat_bottom{
			X_[0].get(),
			H_[0].get()
		};
		const vector<Blob<Dtype>*> concat_top(1, XH_[0].get());
		LayerParameter concat_param;
		concat_param.mutable_concat_param()->set_axis(2);
		concat_.reset(new ConcatLayer<Dtype>(concat_param));
		concat_->SetUp(concat_bottom, concat_top);

		// setup ip_g_ layer
		// Top
		G_.resize(T_);
		for (int t = 0; t < T_; ++t){
			G_[t].reset(new Blob<Dtype>(gate_shape));
		}
		const vector<Blob<Dtype>*> ip_g_bottom(1, XH_[0].get());
		const vector<Blob<Dtype>*> ip_g_top(1, G_[0].get());
		//Layer
		LayerParameter ip_g_param(this->layer_param_);
		ip_g_param.mutable_inner_product_param()->set_num_output(4 * hidden_dim_);
		ip_g_param.mutable_inner_product_param()->set_axis(2);
		ip_g_.reset(new InnerProductLayer<Dtype>(ip_g_param));
		ip_g_->SetUp(ip_g_bottom, ip_g_top);

		blobs_[0].reset(new Blob<Dtype>(ip_h_->blobs()[0]->shape()));
		blobs_[0]->ShareData(*(ip_h_->blobs())[0]);
		blobs_[0]->ShareData(*(ip_h_->blobs())[0]);

		blobs_[1].reset(new Blob<Dtype>(ip_g_->blobs()[0]->shape()));
		blobs_[1]->ShareData(*(ip_g_->blobs())[0]);
		blobs_[1]->ShareDiff(*(ip_g_->blobs())[0]);

		if (bias_term_){
			blobs_[2].reset(new Blob<Dtype>(ip_h_->blobs()[1]->shape()));
			blobs_[2]->ShareData(*(ip_h_->blobs())[1]);
			blobs_[2]->ShareData(*(ip_h_->blobs())[1]);
			blobs_[3].reset(new Blob<Dtype>(ip_g_->blobs()[1]->shape()));
			blobs_[3]->ShareData(*(ip_g_->blobs())[1]);
			blobs_[3]->ShareDiff(*(ip_g_->blobs())[1]);
		}

		// setup dlstm_unit_ layer
		// Bottom
		C_.resize(T_);
		for (int t = 0; t < T_; ++t){
			C_[t].reset(new Blob<Dtype>(h_shape));
		}
		const vector<Blob<Dtype>*> dlstm_unit_bottom{
			C_[0].get(),
			G_[0].get()
		};
		const vector<Blob<Dtype>*> dlstm_unit_top{
			C_[0].get(),
			H_[0].get()
		};
		//Layer
		dlstm_unit_.reset(new DLSTMUnitLayer<Dtype>(LayerParameter()));
		dlstm_unit_->SetUp(dlstm_unit_bottom, dlstm_unit_top);
	}

	template <typename Dtype>
	void DLSTMLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top){
		DRNNBaseLayer<Dtype>::Reshape(bottom, top);
		const vector<int> h_shape{
			1,
			bottom[0]->shape(1),
			hidden_dim_
		};
		// length of sequence has changed
		if (C_.size() != H_.size()){
			C_.resize(T_);
			for (int t = 0; t < T_; ++t){
				C_[t].reset(new Blob<Dtype>(h_shape));
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
			concat_bottom[0] = H0_[seq_id].get();
			concat_bottom[1] = zero_blob_.get();
		}
		else{
			concat_bottom[0] = H_2_[t - 1].get();
			if (conditional_){
				concat_bottom[1] = X_[t - 1].get();
			}
			else{
				concat_bottom[1] = Y_2_[t - 1].get();
			}
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
			dlstm_unit_bottom[0] = C0_[seq_id].get();
		}
		else{
			dlstm_unit_bottom[0] = C_[t - 1].get();
		}
		dlstm_unit_bottom[1] = G_[t].get();
		const vector<Blob<Dtype>*> dlstm_unit_top{
			C_[t].get(),
			H_1_[t].get()
		};
		dlstm_unit_->Forward(dlstm_unit_bottom, dlstm_unit_top);

		// 7. split
		const vector<Blob<Dtype>*> split_h_bottom(1, H_1_[t].get());
		const vector<Blob<Dtype>*> split_h_top{ H_[t].get(), H_2_[t].get() };
		split_h_->Forward(split_h_bottom, split_h_top);
	}

	template <typename Dtype>
	void DLSTMLayer<Dtype>::RecurrentBackward(const int t, const int cont_t,
		const int seq_id){
		// 7. split
		const vector<Blob<Dtype>*> split_h_bottom(1, H_1_[t].get());
		const vector<Blob<Dtype>*> split_h_top{ H_[t].get(), H_2_[t].get() };
		split_h_->Backward(split_h_top,
			vector<bool>(1, true),
			split_h_bottom);

		// 6. dlstm unit
		vector<Blob<Dtype>*> dlstm_unit_bottom(2, NULL);
		if (!cont_t){
			//begin of a sequence
			dlstm_unit_bottom[0] = C0_[seq_id].get();
		}
		else{
			dlstm_unit_bottom[0] = C_[t - 1].get();
		}
		dlstm_unit_bottom[1] = G_[t].get();
		const vector<Blob<Dtype>*> dlstm_unit_top{
			C_[t].get(),
			H_1_[t].get()
		};
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
			concat_bottom[0] = H0_[seq_id].get();
			concat_bottom[1] = zero_blob_.get();
		}
		else{
			concat_bottom[0] = H_2_[t - 1].get();
			if (conditional_){
				concat_bottom[1] = X_[t - 1].get();
			}
			else{
				concat_bottom[1] = Y_2_[t - 1].get();
			}
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