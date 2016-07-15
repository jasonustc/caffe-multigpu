#include<vector>
#include<utility>
#include "caffe/layers/pred_lstm_layer.hpp"

namespace caffe{
	template <typename Dtype>
	void PredLSTMLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top){
		PRNNBaseLayer<Dtype>::LayerSetUp(bottom, top);
		bias_term_ = this->layer_param_.inner_product_param().bias_term();
		if (!bias_term_){
			blobs_.resize(2);
		}
		else{
			blobs_.resize(4);
		}
		vector<int> xh_shape(3, 1);
		xh_shape[1] = bottom[0]->shape(1);
		xh_shape[2] = hidden_dim_ + output_dim_;
		vector<int> gate_shape(3, 1);
		gate_shape[1] = bottom[0]->shape(1);
		gate_shape[2] = hidden_dim_ * 4;

		vector<int> h_shape(3, 1);
		h_shape[1] = bottom[0]->shape(1);
		h_shape[2] = hidden_dim_;

		// concat_ layer
		XH_.resize(T_);
		for (int t = 0; t < T_; ++t){
			XH_[t].reset(new Blob<Dtype>(xh_shape));
		}
		// Layer
		vector<Blob<Dtype>*> concat_bottom(2, NULL);
		concat_bottom[0] = Y_[0].get();
		concat_bottom[1] = H_[0].get();
		const vector<Blob<Dtype>*> concat_top(1, XH_[0].get());
		LayerParameter concat_param;
		concat_param.mutable_concat_param()->set_axis(2);
		concat_.reset(new ConcatLayer<Dtype>(concat_param));
		concat_->SetUp(concat_bottom, concat_top);

		// ip_g_ layer
		G_.resize(T_);
		for (int t = 0; t < T_; ++t){
			G_[t].reset(new Blob<Dtype>(gate_shape));
		}
		const vector<Blob<Dtype>*> ip_g_bottom(1, XH_[0].get());
		const vector<Blob<Dtype>*> ip_g_top(1, G_[0].get());
		// Layer
		LayerParameter ip_g_param(this->layer_param_);
		ip_g_param.mutable_inner_product_param()->set_num_output(4 * hidden_dim_);
		ip_g_param.mutable_inner_product_param()->set_axis(2);
		ip_g_.reset(new InnerProductLayer<Dtype>(ip_g_param));
		ip_g_->SetUp(ip_g_bottom, ip_g_top);

		blobs_[0].reset(new Blob<Dtype>(ip_h_->blobs()[0]->shape()));
		blobs_[0]->ShareData(*(ip_h_->blobs())[0]);
		blobs_[0]->ShareDiff(*(ip_h_->blobs())[0]);
		blobs_[1].reset(new Blob<Dtype>(ip_g_->blobs()[0]->shape()));
		blobs_[1]->ShareData(*(ip_g_->blobs())[0]);
		blobs_[1]->ShareData(*(ip_g_->blobs())[0]);

		if (bias_term_){
			blobs_[2].reset(new Blob<Dtype>(ip_h_->blobs()[1]->shape()));
			blobs_[2]->ShareData(*(ip_h_->blobs())[1]);
			blobs_[2]->ShareData(*(ip_h_->blobs())[1]);
			blobs_[3].reset(new Blob<Dtype>(ip_g_->blobs()[1]->shape()));
			blobs_[3]->ShareData(*(ip_g_->blobs())[1]);
			blobs_[3]->ShareDiff(*(ip_g_->blobs())[1]);
		}

		// LSTM layerT
		C0_.reset(new Blob<Dtype>(h_shape));
		C_.resize(T_);
		for (int t = 0; t < T_; ++t){
			C_[t].reset(new Blob<Dtype>(h_shape));
		}
		vector<Blob<Dtype>*> lstm_unit_bottom(2, NULL);
		lstm_unit_bottom[0] = C_[0].get();
		lstm_unit_bottom[1] = G_[0].get();
		vector<Blob<Dtype>*> lstm_unit_top(2, NULL);
		lstm_unit_top[0] = C_[0].get();
		lstm_unit_top[1] = H_[0].get();
		lstm_unit_.reset(new DLSTMUnitLayer<Dtype>(LayerParameter()));
		lstm_unit_->SetUp(lstm_unit_bottom, lstm_unit_top);
	}

	template <typename Dtype>
	void PredLSTMLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top){
		PRNNBaseLayer<Dtype>::Reshape(bottom, top);
		if (C_.size() != H_.size()){
			vector<int> xh_shape(3, 1);
			xh_shape[1] = bottom[0]->shape(1);
			xh_shape[2] = hidden_dim_ + output_dim_;

			vector<int> gate_shape(3, 1);
			gate_shape[1] = bottom[0]->shape(1);
			gate_shape[2] = hidden_dim_ * 4;

			vector<int> h_shape(3, 1);
			h_shape[1] = bottom[0]->shape(1);
			h_shape[2] = hidden_dim_;

			// concat_ layer
			XH_.resize(T_);
			G_.resize(T_);
			C_.resize(T_);
			for (int t = 0; t < T_; ++t){
				XH_[t].reset(new Blob<Dtype>(xh_shape));
				G_[t].reset(new Blob<Dtype>(gate_shape));
				C_[t].reset(new Blob<Dtype>(h_shape));
			}
		}
	}

	template <typename Dtype>
	void PredLSTMLayer<Dtype>::RecurrentForward(const int t){
		// begin of a sequence
		const bool is_begin = ((t % L_) == 0);
		// 4. concat input_t and h_{t - 1}
		vector<Blob<Dtype>*> concat_bottom(2, NULL);
		if (is_begin){
			const int seq_id = t / L_;
			concat_bottom[0] = start_blob_.get();
			concat_bottom[1] = H0_[seq_id].get();
		}
		else{
			concat_bottom[0] = Y_[t - 1].get();
			concat_bottom[1] = H_[t - 1].get();
		}
		LOG(INFO) << XH_[t]->shape_string();
		vector<Blob<Dtype>*> concat_top(1, XH_[t].get());
		concat_->Forward(concat_bottom, concat_top);

		// 5. forward gate
		const vector<Blob<Dtype>*> ip_g_bottom(1, XH_[t].get());
		const vector<Blob<Dtype>*> ip_g_top(1, G_[t].get());
		ip_g_->Forward(ip_g_bottom, ip_g_top);

		// 6. dlstm unit
		vector<Blob<Dtype>*> lstm_unit_bottom(2, NULL);
		if (is_begin){
			lstm_unit_bottom[0] = C0_.get();
		}
		else{
			lstm_unit_bottom[0] = C_[t - 1].get();
		}
		lstm_unit_bottom[1] = G_[t].get();
		const vector<Blob<Dtype>*> lstm_unit_top{
			C_[t].get(),
			H_[t].get()
		};
		lstm_unit_->Forward(lstm_unit_bottom, lstm_unit_top);
	}

#ifdef CPU_ONLY
	STUB_GPU(PredLSTMLayer);
#endif

	INSTANTIATE_CLASS(PredLSTMLayer);
	REGISTER_LAYER_CLASS(PredLSTM);
}