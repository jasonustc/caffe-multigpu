#include <vector>
#include <utility>

#include "caffe/layers/rbm_layer.hpp"

namespace caffe{

	/*
	 * parameter for inner_product layer is copied from inner_product_param
	 * parameter for sampling is copied from sample_param
	 */
	template <typename Dtype>
	void RBMLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top){
		CHECK_EQ(bottom[0]->num_axes(), 2) << "only blob with 2 dim are allowed as input for rbm";
		int axis = this->layer_param_.inner_product_param().axis();
		axis = bottom[0]->CanonicalAxisIndex(axis);
		CHECK_LE(axis, 1);

		// CD-k 
		num_iter_ = this->layer_param_.rbm_param().num_iteration();
		CHECK_GE(num_iter_, 1) << "iteration times should be at least 1.";

		// setup intermediate data blobs
		vector<int> v_shape = bottom[0]->shape();
		vector<int> h_shape = v_shape;
		h_shape[1] = this->layer_param_.inner_product_param().num_output();
		pos_v_.reset(new Blob<Dtype>(v_shape));
		neg_v_.reset(new Blob<Dtype>(v_shape));
		v_state_.reset(new Blob<Dtype>(v_shape));
		pos_h_.reset(new Blob<Dtype>(h_shape));
		neg_h_.reset(new Blob<Dtype>(h_shape));
		h_state_.reset(new Blob<Dtype>(h_shape));

		// setup forward inner product layer
		// Bottom && Top
		pos_v_->ShareData(*(bottom[0]));
		pos_v_->ShareDiff(*(bottom[0]));
		LayerParameter forward_param(this->layer_param_);
		CHECK(this->layer_param_.inner_product_param().bias_term()) << "bias is required in rbm";
		ip_forward_layer_.reset(new InnerProductLayer<Dtype>(forward_param));
		const vector<Blob<Dtype>*> ip_forward_bottom(1, pos_v_.get());
		const vector<Blob<Dtype>*> ip_forward_top(1, pos_h_.get());
		ip_forward_layer_->SetUp(ip_forward_bottom, ip_forward_top);
        
		// setup hidden activation layer
		// NOTE: here we use the same act layer and sampling layer for 
		// visible and hidden variables, so we only setup them once
		// in-place activation
		act_layer_.reset(new SigmoidLayer<Dtype>(LayerParameter()));
		const vector<Blob<Dtype>*> act_bottom(1, pos_h_.get());
		const vector<Blob<Dtype>*> act_top(1, pos_h_.get());
		act_layer_->SetUp(act_bottom, act_top);

		// setup sampling layer
		sample_layer_.reset(new SamplingLayer<Dtype>(this->layer_param_));
		const vector<Blob<Dtype>*> sample_bottom(1, pos_h_.get());
		const vector<Blob<Dtype>*> sample_top(1, h_state_.get());
		sample_layer_->SetUp(sample_bottom, sample_top);

		// setup backward inner product layer
		LayerParameter back_param(this->layer_param_);
		// for sharing weight with forward_layer_
		back_param.mutable_inner_product_param()->set_transpose(true);
		back_param.mutable_inner_product_param()->set_num_output(v_shape[1]);
		ip_back_layer_.reset(new InnerProductLayer<Dtype>(back_param));
		const vector<Blob<Dtype>*> ip_back_bottom(1, h_state_.get());
		const vector<Blob<Dtype>*> ip_back_top(1, neg_v_.get());
		ip_back_layer_->SetUp(ip_back_bottom, ip_back_top);
		// share parameter
		Blob<Dtype>* back_weight = ip_back_layer_->blobs()[0].get();
		Blob<Dtype>* forward_weight = ip_forward_layer_->blobs()[0].get();
		back_weight->ShareData(*forward_weight);
		back_weight->ShareDiff(*forward_weight);

		// setup learnable params for this layer
		bias_term_ = this->layer_param_.inner_product_param().bias_term();
		this->blobs_.resize(1 + 2 * bias_term_);
		vector<int> forward_w_shape = ip_forward_layer_->blobs()[0]->shape();
		this->blobs_[0].reset(new Blob<Dtype>(forward_w_shape));
		this->blobs_[0]->ShareData(*forward_weight);
		this->blobs_[0]->ShareDiff(*forward_weight);
		if (bias_term_){
			Blob<Dtype>* forward_bias = ip_forward_layer_->blobs()[1].get();
			vector<int> forward_b_shape = forward_bias->shape();
			this->blobs_[1].reset(new Blob<Dtype>(forward_b_shape));
			this->blobs_[1]->ShareData(*forward_bias);
			this->blobs_[1]->ShareDiff(*forward_bias);
			Blob<Dtype>* back_bias = ip_back_layer_->blobs()[1].get();
			vector<int> back_b_shape = back_bias->shape();
			this->blobs_[2].reset(new Blob<Dtype>(back_b_shape));
			this->blobs_[2]->ShareData(*back_bias);
			this->blobs_[2]->ShareDiff(*back_bias);
		}


		// setup top data
		top[0]->ReshapeLike(*(pos_h_.get()));
		top[0]->ShareData(*(pos_h_.get()));

		/// data dim
		// num
		M_ = bottom[0]->count(0, axis);
		// bottom feat dim
		K_ = bottom[0]->count(axis);
		// num_output
		N_ = top[0]->count(axis);

		// bias
		if (bias_term_){
			vector<int> bias_shape(1, M_);
			bias_multiplier_ = new Blob<Dtype>(bias_shape);
			caffe_set(M_, Dtype(1), bias_multiplier_->mutable_cpu_data());
		}

		// weight diff buf
		vector<int> weight_shape(2, N_);
		weight_shape[1] = K_;
		weight_diff_buf_ = new Blob<Dtype>(weight_shape);

		// rbm param update
		this->param_propagate_down_.resize(this->blobs_.size(), true);
	}

	template <typename Dtype>
	void RBMLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top){
		// ip_forward layer
		vector<Blob<Dtype>*> ip_forward_bottom(1, pos_v_.get());
		vector<Blob<Dtype>*> ip_forward_top(1, pos_h_.get());
		ip_forward_layer_->Reshape(ip_forward_bottom, ip_forward_top);
		// sampling layer
		vector<Blob<Dtype>*> sample_bottom(1, pos_h_.get());
		vector<Blob<Dtype>*> sample_top(1, h_state_.get());
		sample_layer_->Reshape(sample_bottom, sample_top);
		// ip_back layer
		const vector<Blob<Dtype>*> ip_back_bottom(1, h_state_.get());
		const vector<Blob<Dtype>*> ip_back_top(1, neg_v_.get());
		ip_back_layer_->Reshape(ip_back_bottom, ip_back_top);
		// sampling layer
		sample_bottom[0] = neg_v_.get();
		sample_top[0] = v_state_.get();
		sample_layer_->Reshape(sample_bottom, sample_top);
		// ip_forward layer
		ip_forward_bottom[0] = v_state_.get();
		ip_forward_top[0] = neg_h_.get();
		ip_forward_layer_->Reshape(ip_forward_bottom, ip_forward_top);
		// top data
		top[0]->ReshapeLike(*(pos_h_.get()));
		top[0]->ShareData(*(pos_h_.get()));
		//output reconstruction loss
		//TODO: output other types of loss, like negative log likelihood
		if (top.size() > 1){
			vector<int> loss_shape(0);
			top[1]->Reshape(loss_shape);
		}
	}

	//CD-k: iterate reconstruction for k times
	// \partial_W = <v_k * h_k> - <v_0 * h_0>
	template <typename Dtype>
	void RBMLayer<Dtype>::Gibbs_vhvh(){
		this->ShareWeight();
		// forward
		vector<Blob<Dtype>*> ip_forward_bottom(1, pos_v_.get());
		vector<Blob<Dtype>*> ip_forward_top(1, pos_h_.get());
		ip_forward_layer_->Forward(ip_forward_bottom, ip_forward_top);
		// sigmoid activation
		vector<Blob<Dtype>*> act_bottom(1, pos_h_.get());
		vector<Blob<Dtype>*> act_top(1, pos_h_.get());
		act_layer_->Forward(act_bottom, act_top);
		// sampling
		vector<Blob<Dtype>*> sample_bottom(1, pos_h_.get());
		vector<Blob<Dtype>*> sample_top(1, h_state_.get());
		sample_layer_->Forward(sample_bottom, sample_top);
		// back
		vector<Blob<Dtype>*> ip_back_bottom(1, h_state_.get());
		vector<Blob<Dtype>*> ip_back_top(1, neg_v_.get());
		ip_back_layer_->Forward(ip_back_bottom, ip_back_top);
		// sigmoid activation
		act_bottom[0] = neg_v_.get();
		act_top[0] = neg_v_.get();
		act_layer_->Forward(act_bottom, act_top);
		// sampling
		sample_bottom[0] = neg_v_.get();
		sample_top[0] = v_state_.get();
		sample_layer_->Forward(sample_bottom, sample_top);
		// forward again
		ip_forward_bottom[0] = v_state_.get();
		ip_forward_top[0] = neg_h_.get();
		ip_forward_layer_->Forward(ip_forward_bottom, ip_forward_top);
		// sigmoid activation
		act_bottom[0] = neg_h_.get();
		act_top[0] = neg_h_.get();
		act_layer_->Forward(act_bottom, act_top);
		for (int k = 1; k < num_iter_; ++k){
			// sampling
			sample_bottom[0] = neg_h_.get();
			sample_top[0] = h_state_.get();
			sample_layer_->Forward(sample_bottom, sample_top);
			// back
			ip_back_bottom[0] = h_state_.get();
			ip_back_top[0] = neg_v_.get();
			ip_back_layer_->Forward(ip_back_bottom, ip_back_top);
			// sigmoid activation
			act_bottom[0] = neg_v_.get();
			act_top[0] = neg_v_.get();
			act_layer_->Forward(act_bottom, act_top);
			// sampling
			sample_bottom[0] = neg_v_.get();
			sample_top[0] = v_state_.get();
			sample_layer_->Forward(sample_bottom, sample_top);
			// forward again
			ip_forward_bottom[0] = v_state_.get();
			ip_forward_top[0] = neg_h_.get();
			ip_forward_layer_->Forward(ip_forward_bottom, ip_forward_top);
			// sigmoid activation
			act_bottom[0] = neg_h_.get();
			act_top[0] = neg_h_.get();
			act_layer_->Forward(act_bottom, act_top);
		}
	}

	template <typename Dtype>
	void RBMLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top){
		Gibbs_vhvh();
		const Dtype* bottom_data = bottom[0]->cpu_data();
		if (top.size() > 1){
			const int count = bottom[0]->count();
			//use neg_v diff for buffer of reconstruction error data
			caffe_sub<Dtype>(count, bottom_data, neg_v_->cpu_data(), neg_v_->mutable_cpu_diff());
			Dtype loss = caffe_cpu_dot<Dtype>(count, neg_v_->cpu_diff(), neg_v_->cpu_diff());
			top[1]->mutable_cpu_data()[0] = loss / bottom[0]->num();
		}
	}

	template <typename Dtype>
	void RBMLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down,
		const vector<Blob<Dtype>*>& bottom){
		//put positive data into buf data
		Dtype* pos_ass_data = weight_diff_buf_->mutable_cpu_data();
		//put negative data into buf diff
		Dtype* neg_ass_data = weight_diff_buf_->mutable_cpu_diff();
		const Dtype* pos_v_data = bottom[0]->cpu_data();
		const Dtype* pos_h_data = pos_h_->cpu_data();
		const Dtype* neg_v_data = neg_v_->cpu_data();
		const Dtype* neg_h_data = neg_h_->cpu_data();
		const Dtype* top_diff = top[0]->cpu_diff();
		Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
		Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();
		const Dtype* weight_data = this->blobs_[0]->cpu_data();
		Dtype scale = Dtype(1.) / Dtype(M_);

		//Gradient with respect to weight
		if (this->param_propagate_down_[0]){
			caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, N_, K_, M_, (Dtype)1.,
				pos_h_data, pos_v_data, (Dtype)0., pos_ass_data);
			caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, N_, K_, M_, (Dtype)1.,
				neg_h_data, neg_v_data, (Dtype)0., neg_ass_data);
			caffe_sub<Dtype>(N_ * K_, pos_ass_data, neg_ass_data, neg_ass_data);
			//average by batch size
			caffe_cpu_axpby<Dtype>(this->blobs_[0]->count(), scale, neg_ass_data,
				Dtype(1.), weight_diff);
		}

		//Gradient with respect to h_bias
		//\delta c_j = \delta c_j + p_h_j^(0) - p_h_j^(k)
		if (bias_term_ && this->param_propagate_down_[1]){
			const int count_h = pos_h_->count();
			Dtype* h_bias_diff = this->blobs_[1]->mutable_cpu_diff();
			//put buffer data in neg_h_.diff()
			//pos_h_ is shared with top[0], be carefully to use it in other place
			caffe_sub<Dtype>(count_h, pos_h_data, neg_h_data, neg_h_->mutable_cpu_diff());
			//average by batch size
			//(M_, N_) here is the raw rows and cols of A
			caffe_cpu_gemv<Dtype>(CblasTrans, M_, N_, scale, neg_h_->cpu_diff(),
				bias_multiplier_->cpu_data(), (Dtype)1., h_bias_diff);
		}

		//Gradient with respect to v_bias
		//\delta b_j = \delta b_j + v_j^(0) - v_j^(k)
		if (bias_term_ && this->param_propagate_down_[2]){
			const int count_v = pos_v_->count();
			Dtype* v_bias_diff = this->blobs_[2]->mutable_cpu_diff();
			//put buffer data in neg_v_.diff()
			//pos_v_ is shared with bottom[0], be carefully to use it in other place
			caffe_sub<Dtype>(count_v, pos_v_data, neg_v_data, neg_v_->mutable_cpu_diff());
			//put intemidiate result into neg_v_ data()
			//average by batch size
			//(M_, K_) here is the raw rows and cols of A
			caffe_cpu_gemv<Dtype>(CblasTrans, M_, K_, scale, neg_v_->cpu_diff(),
				bias_multiplier_->cpu_data(), (Dtype)1., v_bias_diff);
		}

		//Gradient with respect to bottom data
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

#ifdef CPU_ONLY
	STUB_GPU(RBMLayer);
#endif

	INSTANTIATE_CLASS(RBMLayer);
	REGISTER_LAYER_CLASS(RBM);
} // namespace caffe
