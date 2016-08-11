#include <string>
#include <utility>
#include <vector>

#include "caffe/layers/lstm_layer.hpp"

namespace caffe {

	/*
	 * bottom[0]->shape(1): #streams
	 */
	template <typename Dtype>
	void LSTMLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top)
	{
		RNNBaseLayer<Dtype>::LayerSetUp(bottom, top);

		this->bias_term_ = this->layer_param_.inner_product_param().bias_term();
		out_ct_ = (top.size() > 1);
		if (!this->bias_term_)
		{
			this->blobs_.resize(1);
		}
		else
		{
			this->blobs_.resize(2);
		}

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
		h_shape[2] = this->hidden_dim_;
    /*
		const vector<int> xh_shape {
			1,
			bottom[0]->shape(1),
			bottom[0]->shape(2) + hidden_dim_
		};*/
		vector<int> xh_shape(3, 1);
		xh_shape[1] = bottom[0]->shape(1);
		xh_shape[2] = bottom[0]->shape(2) + this->hidden_dim_;
    /*
		const vector<int> gate_shape{
			1,
			bottom[0]->shape(1),
			hidden_dim_ * 4
		};*/
		vector<int> gate_shape(3, 1);
		gate_shape[1] = bottom[0]->shape(1);
		gate_shape[2] = this->hidden_dim_ * 4;
    /*
		const vector<int> cont_shape{
			1,
			bottom[0]->shape(1)
		};*/
		vector<int> cont_shape(2, 1);
		cont_shape[1] = bottom[0]->shape(1);

		// setup split_h_ layer
		// Bottom & Top
		H_1_.resize(this->T_);
		H_2_.resize(this->T_);
		for (int t = 0; t < this->T_; ++t)
		{
			H_1_[t].reset(new Blob<Dtype>(h_shape));
			H_2_[t].reset(new Blob<Dtype>(h_shape));
		}
		H0_.reset(new Blob<Dtype>(h_shape));
		// Layer
		const vector<Blob<Dtype>*> split_h_bottom(1, H_1_[0].get());
		const vector<Blob<Dtype>*> split_h_top(2, H_2_[0].get());
		split_h_.reset(new SplitLayer<Dtype>(LayerParameter()));
		split_h_->SetUp(split_h_bottom, split_h_top);

		// setup scale_h_ layer
		// Top
		SH_.resize(this->T_);
		for (int t = 0; t < this->T_; ++t)
		{
			SH_[t].reset(new Blob<Dtype>(h_shape));
		}
		// Layer
		// element-wise multiplication
    /*
		const vector<Blob<Dtype>*> scale_h_bottom{
			this->H_[0].get(),
			this->CONT_[0].get()
		};*/
		vector<Blob<Dtype>*> scale_h_bottom(2, NULL);
		scale_h_bottom[0] = this->H_[0].get();
		scale_h_bottom[1] = this->CONT_[0].get();
		const vector<Blob<Dtype>*> scale_h_top(1, SH_[0].get());
		LayerParameter scale_param;
		scale_param.mutable_scale_param()->set_axis(0);
		scale_h_.reset(new ScaleLayer<Dtype>(scale_param));
		scale_h_->SetUp(scale_h_bottom, scale_h_top);

		// setup concat_h_ layer
		// Bottom & Top
		XH_.resize(this->T_);
		for (int t = 0; t < this->T_; ++t)
		{
			XH_[t].reset(new Blob<Dtype>(xh_shape));
		}
		// Layer
    /*
		const vector<Blob<Dtype>*> concat_bottom {
			X_[0].get(),
			H_[0].get()
		};*/
		vector<Blob<Dtype>*> concat_bottom(2, NULL);
		concat_bottom[0] = this->X_[0].get();
		concat_bottom[1] = this->H_[0].get();
		const vector<Blob<Dtype>*> concat_top(1, XH_[0].get());
		LayerParameter concat_param;
		concat_param.mutable_concat_param()->set_axis(2);
		concat_.reset(new ConcatLayer<Dtype>(concat_param));
		concat_->SetUp(concat_bottom, concat_top);

		//setup ip_g_ layer
		// Top
		G_.resize(this->T_);
		for (int t = 0; t < this->T_; ++t)
		{
			G_[t].reset(new Blob<Dtype>(gate_shape));
		}
		// Layer
		const vector<Blob<Dtype>*> ip_g_bottom(1, XH_[0].get());
		const vector<Blob<Dtype>*> ip_g_top(1, G_[0].get());
		LayerParameter ip_param;
		ip_param.mutable_inner_product_param()->CopyFrom(
			this->layer_param().inner_product_param());
		ip_param.mutable_inner_product_param()->set_num_output(this->hidden_dim_ * 4);
		ip_param.mutable_inner_product_param()->set_axis(2);
		ip_g_.reset(new InnerProductLayer<Dtype>(ip_param));
		ip_g_->SetUp(ip_g_bottom, ip_g_top);
		
		this->blobs_[0].reset(new Blob<Dtype>(ip_g_->blobs()[0]->shape()));
		this->blobs_[0]->ShareData(*(ip_g_->blobs())[0]);
		this->blobs_[0]->ShareDiff(*(ip_g_->blobs())[0]);
		if (this->bias_term_)
		{
		this->blobs_[1].reset(new Blob<Dtype>(ip_g_->blobs()[1]->shape()));
		this->blobs_[1]->ShareData(*(ip_g_->blobs())[1]);
		this->blobs_[1]->ShareDiff(*(ip_g_->blobs())[1]);
		}
		// setup lstm_unit_h_ layer
		// Bottom
		C_.resize(this->T_);
		for (int t = 0; t < this->T_; ++t)
		{
			C_[t].reset(new Blob<Dtype>(h_shape));
		}
		C0_.reset(new Blob<Dtype>(h_shape));
		// Layer
    /*
		vector<Blob<Dtype>*> lstm_unit_bottom {
			C_[0].get(),
			G_[0].get(),
			CONT_[0].get()
		};*/
		vector<Blob<Dtype>*> lstm_unit_bottom(3, NULL);
		lstm_unit_bottom[0] = C_[0].get();
		lstm_unit_bottom[1] = G_[0].get();
		lstm_unit_bottom[2] = this->CONT_[0].get();
    /*
		vector<Blob<Dtype>*> lstm_unit_top{
			C_[0].get(),
			H_[0].get()
		};*/
		vector<Blob<Dtype>*> lstm_unit_top(2, C_[0].get());
		lstm_unit_top[1] = this->H_[0].get();
		lstm_unit_.reset(new LSTMUnitLayer<Dtype>(LayerParameter()));
		lstm_unit_->SetUp(lstm_unit_bottom, lstm_unit_top);

		// setup split_c_ layer
		// Top
		C_1_.resize(this->T_);
		for (int t = 0; t < this->T_; ++t){
			C_1_[t].reset(new Blob<Dtype>(h_shape));
		}
		if (out_ct_){
			C_2_.resize(this->T_);
			for (int t = 0; t < this->T_; ++t){
				C_2_[t].reset(new Blob<Dtype>(h_shape));
			}
			const vector<Blob<Dtype>*> split_c_bottom(1, C_[0].get());
      /*
			const vector<Blob<Dtype>*> split_c_top{
				C_1_[0].get(),
				C_2_[0].get()
			};*/
			vector<Blob<Dtype>*> split_c_top(2, C_1_[0].get());
			split_c_top[1] = C_2_[0].get();
			//Layer
			split_c_.reset(new SplitLayer<Dtype>(LayerParameter()));
			split_c_->SetUp(split_c_bottom, split_c_top);
		}
		else{
			for (int t = 0; t < this->T_; ++t){
				C_1_[t]->ShareData(*(C_[t].get()));
				C_1_[t]->ShareDiff(*(C_[t].get()));
			}
		}
	}

	template <typename Dtype>
	void LSTMLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top){
		RNNBaseLayer<Dtype>::Reshape(bottom, top);
		//length of the sequence has changed
		if (this->H_.size() != C_.size()){
      /*
			const vector<int> h_shape{
				1,
				bottom[0]->shape(1),
				hidden_dim_
			};*/
			vector<int> h_shape(3, 1);
			h_shape[1] = bottom[0]->shape(1);
			h_shape[2] = this->hidden_dim_;
			C_.resize(this->T_);
			C_1_.resize(this->T_);
			for (int t = 0; t < this->T_; ++t)
			{
				C_[t].reset(new Blob<Dtype>(h_shape));
				C_1_[t].reset(new Blob<Dtype>(h_shape));
			}
			if (out_ct_){
				C_2_.resize(this->T_);
				for (int t = 0; t < this->T_; ++t){
					C_2_[t].reset(new Blob<Dtype>(h_shape));
				}
			}
			else{
				for (int t = 0; t < this->T_; ++t){
					C_1_[t]->ShareData(*(C_[t].get()));
					C_1_[t]->ShareDiff(*(C_[t].get()));
				}
			}
		}// if (H_.size() != C_.size())
	}

	template <typename Dtype>
	void LSTMLayer<Dtype>::RecurrentForward(const int t) {
			//3. scale_h_
			vector<Blob<Dtype>*> scale_h_bottom(2, NULL);
			if (t == 0)
			{
				scale_h_bottom[0] = H0_.get();
			}
			else
			{
				scale_h_bottom[0] = H_2_[t - 1].get();
			}
			scale_h_bottom[1] = this->CONT_[t].get();
			const vector<Blob<Dtype>*> scale_h_top(1, SH_[t].get());
			scale_h_->Forward(scale_h_bottom, scale_h_top);

			//4. concat x_t & h_t-1.
      /*
			vector<Blob<Dtype>*> concat_bottom{
				X_[t].get(),
				SH_[t].get()
			};*/
			vector<Blob<Dtype>*> concat_bottom(2, NULL);
			concat_bottom[0] = this->X_[t].get();
			concat_bottom[1] = SH_[t].get();
			const vector<Blob<Dtype>*> concat_top(1, XH_[t].get());
			concat_->Forward(concat_bottom, concat_top);

			//5. forward gate.
			const vector<Blob<Dtype>*> ip_g_bottom(1, XH_[t].get());
			const vector<Blob<Dtype>*> ip_g_top(1, G_[t].get());
			ip_g_->Forward(ip_g_bottom, ip_g_top);

			//6. LSTM Unit.
			vector<Blob<Dtype>*> lstm_bottom(3, NULL);
			if (t == 0)
			{
				lstm_bottom[0] = C0_.get();
			}
			else
			{
				lstm_bottom[0] = C_1_[t - 1].get();
			}
			lstm_bottom[1] = G_[t].get();
			lstm_bottom[2] = this->CONT_[t].get();
      /*
			vector<Blob<Dtype>*> lstm_top{
				C_[t].get(),
				H_1_[t].get()
			};*/
			vector<Blob<Dtype>*> lstm_top(2, C_[t].get());
			lstm_top[1] = H_1_[t].get();
			lstm_unit_->Forward(lstm_bottom, lstm_top);
			//7. split
			const vector<Blob<Dtype>*> split_h_bottom(1, H_1_[t].get());
			/*const vector<Blob<Dtype>*> split_h_top{ this->H_[t].get(), H_2_[t].get() };*/
			vector<Blob<Dtype>*> split_h_top(2, this->H_[t].get()); 
      split_h_top[1] = H_2_[t].get();
			split_h_->Forward(split_h_bottom, split_h_top);
			if (out_ct_){
				const vector<Blob<Dtype>*> split_c_bottom(1, C_[t].get());
        /*
				const vector<Blob<Dtype>*> split_c_top{
					C_1_[t].get(),
					C_2_[t].get()
				};*/
				vector<Blob<Dtype>*> split_c_top(2, C_1_[t].get());
				split_c_top[1] = C_2_[t].get();
				split_c_->Forward(split_c_bottom, split_c_top);
			}
	}

	template <typename Dtype>
	void LSTMLayer<Dtype>::RecurrentBackward(const int t) {
			//7. split
			const vector<Blob<Dtype>*> split_h_bottom(1, H_1_[t].get());
			/*const vector<Blob<Dtype>*> split_h_top{ this->H_[t].get(), H_2_[t].get() };*/
			vector<Blob<Dtype>*> split_h_top(2, this->H_[t].get());
      split_h_top[1] = H_2_[t].get();
			split_h_->Backward(split_h_top,
				vector<bool>(1, true),
				split_h_bottom);
			if (out_ct_){
				const vector<Blob<Dtype>*> split_c_bottom(1, C_[t].get());
        /*
				const vector<Blob<Dtype>*> split_c_top{
					C_1_[t].get(),
					C_2_[t].get()
				};*/
				vector<Blob<Dtype>*> split_c_top(2, C_1_[t].get());
				split_c_top[1] = C_2_[t].get();
				split_c_->Backward(split_c_top,
					vector<bool>(1, true),
					split_c_bottom);
			}

			//6. LSTM Unit.
			vector<Blob<Dtype>*> lstm_bottom(3, NULL);
			if (t == 0)
			{
				lstm_bottom[0] = C0_.get();
			}
			else
			{
				lstm_bottom[0] = C_1_[t - 1].get();
			}
			lstm_bottom[1] = G_[t].get();
			lstm_bottom[2] = this->CONT_[t].get();
      /*
			vector<Blob<Dtype>*> lstm_top{
				C_[t].get(),
				H_1_[t].get()
			};*/
			vector<Blob<Dtype>*> lstm_top(2, C_[t].get());
			lstm_top[1] = H_1_[t].get();
			/*const vector<bool> lstm_unit_bool{ true, true, false };*/
			vector<bool> lstm_unit_bool(3, true);
      lstm_unit_bool[2] = false;
			lstm_unit_->Backward(lstm_top,
				lstm_unit_bool,
				lstm_bottom);

			//5. forward gate.
			const vector<Blob<Dtype>*> ip_g_bottom(1, XH_[t].get());
			const vector<Blob<Dtype>*> ip_g_top(1, G_[t].get());
			ip_g_->Backward(ip_g_top,
				vector<bool>(1, true),
				ip_g_bottom);

			//4. concat x_t & h_t-1.
      /*
			vector<Blob<Dtype>*> concat_bottom{
				X_[t].get(),
				SH_[t].get()
			};*/
			vector<Blob<Dtype>*> concat_bottom(2, NULL);
			concat_bottom[0] = this->X_[t].get();
			concat_bottom[1] = SH_[t].get();
			const vector<Blob<Dtype>*> concat_top(1, XH_[t].get());
			concat_->Backward(concat_top,
				vector<bool>(2, true),
				concat_bottom);

			//3. scale_h_
			vector<Blob<Dtype>*> scale_h_bottom(2, NULL);
			if (t == 0)
			{
				scale_h_bottom[0] = H0_.get();
			}
			else
			{
				scale_h_bottom[0] = H_2_[t - 1].get();
			}
			scale_h_bottom[1] = this->CONT_[t].get();
			const vector<Blob<Dtype>*> scale_h_top(1, SH_[t].get());
			/*const vector<bool> scale_h_bool{ true, false };*/
			vector<bool> scale_h_bool(2, true);
      scale_h_bool[1] = false;
			scale_h_->Backward(scale_h_top,
				scale_h_bool,
				scale_h_bottom);
	}

	template <typename Dtype>
	void LSTMLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top){
		RNNBaseLayer<Dtype>::Forward_cpu(bottom, top);
		// 10. concat_ct_
		if (out_ct_){
			vector<Blob<Dtype>*> concat_ct_bottom(this->T_, NULL);
			for (int t = 0; t < this->T_; ++t){
				concat_ct_bottom[t] = C_2_[t].get();
			}
			const vector<Blob<Dtype>*> concat_ct_top(1, top[1]);
			concat_ct_->Forward(concat_ct_bottom, concat_ct_top);
		}
	}

	template <typename Dtype>
	void LSTMLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom){
		if (out_ct_){
			// 10. concat_ct_
			vector<Blob<Dtype>*> concat_ct_bottom(this->T_, NULL);
			for (int t = 0; t < this->T_; ++t){
				concat_ct_bottom[t] = C_2_[t].get();
			}
			const vector<Blob<Dtype>*> concat_ct_top(1, top[1]);
			concat_ct_->Backward(concat_ct_top,
				vector<bool>(this->T_, true),
				concat_ct_bottom);
		}
		RNNBaseLayer<Dtype>::Backward_cpu(top, propagate_down, bottom);
	}

#ifdef CPU_ONLY
	STUB_GPU(LSTMLayer);
#endif

	INSTANTIATE_CLASS(LSTMLayer);
	REGISTER_LAYER_CLASS(LSTM);
}  // namespace caffe
