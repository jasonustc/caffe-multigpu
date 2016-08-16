#include <vector>
#include <utility>

#include "caffe/layers/local_lstm_layer.hpp"

namespace caffe{
	template <typename Dtype>
	void LocalLSTMLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top){
		LSTMLayer<Dtype>::LayerSetUp(bottom, top);
		vector<int> x_shape(3, 1);
		x_shape[1] = bottom[0]->shape(1);
		x_shape[2] = bottom[0]->shape(2);
		local_bias_term_ = this->layer_param_.recurrent_param().local_bias_term();
		LOG(INFO) << "local_bias_term: " << local_bias_term_;
		// predict of x before and after activation
		xp_.reset(new Blob<Dtype>(x_shape));
		xp_act_.reset(new Blob<Dtype>(x_shape));

		// setup predict inner_product layer
		LayerParameter ip_xp_param;
		// filler setting
		ip_xp_param.CopyFrom(this->layer_param_.inner_product_param());
		// axis and num_output
		ip_xp_param.mutable_inner_product_param()->set_axis(2);
		ip_xp_param.mutable_inner_product_param()->set_num_output(x_shape[2]);
		ip_xp_param.mutable_inner_product_param()->set_bias_term(local_bias_term_);
		ip_xp_.reset(new InnerProductLayer<Dtype>(ip_xp_param));
		const vector<Blob<Dtype>*> ip_xp_bottom(1, this->H_[0].get());
		const vector<Blob<Dtype>*> ip_xp_top(1, xp_.get());
		ip_xp_->SetUp(ip_xp_bottom, ip_xp_top);

		// setup activation layer
		switch (this->layer_param_.recurrent_param().local_act_type()){
		case RecurrentParameter_ActType_RELU:
			act_layer_.reset(new ReLULayer<Dtype>(LayerParameter()));
			break;
		case RecurrentParameter_ActType_SIGMOID:
			act_layer_.reset(new SigmoidLayer<Dtype>(LayerParameter()));
			break;
		default:
			LOG(FATAL) << "Unkown activation type";
		}
		// NOTE: maybe a in-place operation is enough
		const vector<Blob<Dtype>*> act_bottom(1, xp_.get());
		const vector<Blob<Dtype>*> act_top(1, xp_act_.get());
		act_layer_->SetUp(act_bottom, act_top);

		// setup local loss layer
		// TODO: allow more types of loss layers
		LayerParameter local_loss_param;
		local_loss_param.set_loss_weight(
			this->layer_param_.recurrent_param().local_loss_weight());
		loss_layer_.reset(new EuclideanLossLayer<Dtype>(local_loss_param));
		vector<Blob<Dtype>*> local_loss_bottom(2, NULL);
		local_loss_bottom[0] = xp_act_.get();
		local_loss_bottom[1] = this->X_[0].get();
		local_loss_.reset(new Blob<Dtype>());
		const vector<Blob<Dtype>*> local_loss_top(1, local_loss_.get());
		loss_layer_->SetUp(local_loss_bottom, local_loss_top);
		
		// local learning parameters
		local_lr_ = this->layer_param_.recurrent_param().local_lr();
		CHECK_GE(local_lr_, 0);
		LOG(INFO) << "local learning rate: " << local_lr_;
		local_decay_ = this->layer_param_.recurrent_param().local_decay();
		CHECK_GE(local_decay_, 0);
		LOG(INFO) << "lcoal decay: " << local_decay_;
		local_gradient_clip_ = this->layer_param_.recurrent_param().local_gradient_clip();
		CHECK_GE(local_gradient_clip_, 0);
		local_momentum_ = this->layer_param_.recurrent_param().local_momentum();

		// TODO: check if bias should be included in learnable parameters
		// local learn parameters
		// 1. LSTM parameters
		local_learn_params_.push_back(this->blobs()[0]);
		if (this->bias_term_){
			local_learn_params_.push_back(this->blobs()[1]);
		}
		// 2. local inner_product parameters
		local_learn_params_.push_back(ip_xp_->blobs()[0]);
		if (local_bias_term_){
			local_learn_params_.push_back(ip_xp_->blobs()[1]);
		}
		
		// temp_ for L1 decay
		for (int i = 0; i < local_learn_params_.size(); ++i){
			temp_.push_back(new Blob<Dtype>(local_learn_params_[i]->shape()));
		}
		CHECK_EQ(this->layer_param_.recurrent_param().local_param_size(), local_learn_params_.size()) <<
			"param spec should be set for every parameter";
	}

	template <typename Dtype>
	void LocalLSTMLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top){
		LSTMLayer<Dtype>::Reshape(bottom, top);
		const vector<Blob<Dtype>*> ip_xp_bottom(1, this->H_[0].get());
		const vector<Blob<Dtype>*> ip_xp_top(1, xp_.get());
		ip_xp_->Reshape(ip_xp_bottom, ip_xp_top);
		temp_[0]->Reshape(ip_xp_->blobs()[0]->shape());
		if (bias_term_){
			temp_[1]->Reshape(ip_xp_->blobs()[1]->shape());
		}
		vector<Blob<Dtype>*> local_loss_bottom(2, NULL);
		local_loss_bottom[0] = xp_act_.get();
		local_loss_bottom[1] = this->X_[0].get();
		const vector<Blob<Dtype>*> local_loss_top(1, local_loss_.get());
		loss_layer_->Reshape(local_loss_bottom, local_loss_top);
		// since reshape is called in every forward process, we clear history 
		// diff of local learn parameters here
		for (int i = 0; i < local_learn_params_.size(); ++i){
			switch (Caffe::mode()){
			case Caffe::CPU:{
				caffe_set<Dtype>(local_learn_params_[i]->count(),
					Dtype(0), temp_[i]->mutable_cpu_diff());
				break;
			}
			case Caffe::GPU:{
				caffe_gpu_set<Dtype>(local_learn_params_[i]->count(),
					Dtype(0), temp_[i]->mutable_gpu_diff());
				break;
			}
			default:
				LOG(FATAL) << "Unkown caffe mode: " << Caffe::mode();
			}
		}
	}

	template <typename Dtype>
	void LocalLSTMLayer<Dtype>::Regularize(const Dtype local_decay, const int id){
		Blob<Dtype>* blob = local_learn_params_[id].get();
		string regularize_type = this->layer_param_.recurrent_param().weight_decay_type();
		switch (Caffe::mode()){
		case Caffe::CPU:{
			if (local_decay){
				if (regularize_type == "L2"){
					caffe_axpy(blob->count(), local_decay,
						blob->cpu_data(), blob->mutable_cpu_diff());
				}
				else if (regularize_type == "L1"){
					caffe_cpu_sign(blob->count(), blob->cpu_data(), temp_[id]->mutable_cpu_data());
					caffe_axpy(blob->count(), local_decay,
						temp_[id]->cpu_data(), blob->mutable_cpu_diff());
				}
				else{
					LOG(FATAL) << "unkown regularize type: " << regularize_type;
				}
			}
			break;
		}
		case Caffe::GPU:{
			if (local_decay){
#ifndef CPU_ONLY
				if (regularize_type == "L2"){
					caffe_gpu_axpy(blob->count(), local_decay,
						blob->gpu_data(), blob->mutable_gpu_diff());
				}
				else if (regularize_type == "L1"){
					caffe_gpu_sign(blob->count(), blob->gpu_data(), temp_[id]->mutable_gpu_data());
					caffe_axpy(blob->count(), local_decay,
						temp_[id]->gpu_data(), blob->mutable_gpu_diff());
				}
				else{
					LOG(FATAL) << "unkown regularize type: " << regularize_type;
				}
			}
#else
	NO_GPU;
#endif
			break;
		}
		default:
			LOG(FATAL) << "Unkown caffe mode: " << Caffe::mode();
		}
	}

	template <typename Dtype>
	void LocalLSTMLayer<Dtype>::ClipGradients(){
		if (local_gradient_clip_ <= 0) { return; }
		Dtype sumsq_diff = 0;
		for (int i = 0; i < local_learn_params_.size(); ++i){
			sumsq_diff += local_learn_params_[i]->sumsq_diff();
		}
		const Dtype l2norm_diff = std::sqrt(sumsq_diff);
		if (l2norm_diff > local_gradient_clip_){
			Dtype scale_factor = local_gradient_clip_ / l2norm_diff;
			LOG(INFO) << "Gradient clipping: scaling down gradients (L2 norm"
				<< l2_norm_diff << " > " << local_gradient_clip_ << ") "
				<< "by scale factor " << scale_factor;
			for (int i = 0; i < local_learn_params_.size(); ++i){
				local_learn_params_[i]->scale_diff(scale_factor);
			}
		}
	}

#ifndef CPU_ONLY
	template <typename Dtype>
	void sgd_update_gpu(int N, Dtype* g, Dtype* h, Dtype momentum,
		Dtype local_rate);
#endif
	template <typename Dtype>
	void LocalLSTMLayer<Dtype>::ComputeUpdateValue(const Dtype lr, const Dtype mom, 
		const int id){
		switch (Caffe::mode()){
		case Caffe::CPU:{
			// use temp_->c/gpu_diff() to store history
			caffe_cpu_axpby(local_learn_params_[id]->count(), lr,
				local_learn_params_[id]->cpu_diff(), mom,
				temp_[id]->mutable_cpu_diff());
			caffe_copy(local_learn_params_[id]->count(),
				temp_[id]->mutable_cpu_diff(),
				local_learn_params_[id]->mutable_cpu_diff());
			break;
		}
		case Caffe::GPU:{
#ifndef CPU_ONLY
			sgd_update_gpu(local_learn_params_[id]->count(),
				local_learn_params_[id]->mutable_gpu_diff(),
				temp_[id]->mutable_gpu_diff(),
				mom, lr);
#else 
			NO_GPU;
#endif
	break;
		}
		default:
			LOG(FATAL) << "Unknown caffe mode: " << Caffe::mode();
		}
	}

	template <typename Dtype>
	void LocalLSTMLayer<Dtype>::LocalUpdateRecurrent(){
		// clip gradients
		ClipGradients();
		const int n = local_learn_params_.size();
		// regularize
		for (int i = 0; i < n; ++i){
			ParamSpec* param_spec = this->layer_param_.recurrent_param().local_param(i);
			Dtype local_decay = local_decay_ * param_spec->decay_mult();
			Regularize(local_decay, i);
			Dtype local_lr = local_lr_ * param_spec->lr_mult();
			ComputeUpdateValue(local_lr, local_momentum, i);
		}
		//update
		for (int i = 0; i < n; ++i){
			local_learn_params_[i]->Update();
		}
	}

	template <typename Dtype>
	void LocalLSTMLayer<Dtype>::RecurrentForward(const int t){
		LSTMLayer<Dtype>::RecurrentForward(t);
		// update by prediction
		if (t < this->T_ - 1){
			// innerproduct
			const vector<Blob<Dtype>*> ip_xp_bottom(1, this->H_[t].get());
			const vector<Blob<Dtype>*> ip_xp_top(1, xp_.get());
			ip_xp_->Forward(ip_xp_bottom, ip_xp_top);
			// local loss
			vector<Blob<Dtype>*> local_loss_bottom(2, NULL);
			local_loss_bottom[0] = xp_act_.get();
			local_loss_bottom[1] = this->X_[t + 1].get();
			const vector<Blob<Dtype>*> local_loss_top(1, local_loss_.get());
			loss_layer_->Forward(local_loss_bottom, local_loss_top);
			DLOG(INFO) << "local loss " << t << " : " << local_loss_->cpu_data()[0];
			vector<bool> propagate_down(2, false);
			propagate_down[0] = true;
			loss_layer_->Backward(local_loss_top,
				propagate_down,
				local_loss_bottom);
			ip_xp_->Backward(ip_xp_top,
				vector<bool>(1, true),
				ip_xp_bottom);
			this->RecurrentBackward(t);
			LocalUpdateRecurrent();
		}
	}

	// TODO: implement another alternative: do not backward through time

#ifdef CPU_ONLY
	STUB_GPU(LocalLSTMLayer);
#endif

	INSTANTIATE_CLASS(LocalLSTMLayer);
	REGISTER_LAYER_CLASS(LocalLSTM);
} // namespace caffe