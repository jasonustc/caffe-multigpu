#include <vector>
#include <string>

#include "caffe/layers/adaptive_dropout_layer.hpp"

namespace caffe{
	template <typename Dtype>
	//inline: only self and friend class can call
	inline Dtype sigmoid_cpu(Dtype x){
		return 1. / (1. + exp(-x));
	}

	template <typename Dtype>
	inline Dtype relu_cpu(Dtype x){
		return x > 0 ? x : Dtype(0);
	}

	template <typename Dtype>
	inline void activate_cpu(const int n, const Dtype* in, Dtype* out, 
		AdaptiveDropoutParameter_ActType act_type){
		switch (act_type){
		case caffe::AdaptiveDropoutParameter_ActType_SIGMOID:
			for (int i = 0; i < n; i++){
				out[i] = sigmoid_cpu<Dtype>(in[i]);
			}
			break;
		case caffe::AdaptiveDropoutParameter_ActType_RELU:
			for (int i = 0; i < n; i++){
				out[i] = relu_cpu<Dtype>(in[i]);
			}
			break;
		default:
			LOG(FATAL) << "Unkown activate function.";
		}
	}

	template <typename Dtype>
	void AdaptiveDropoutLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		alpha_ = this->layer_param_.adaptive_dropout_param().alpha();
		beta_ = this->layer_param_.adaptive_dropout_param().beta();
		const int num_output = this->layer_param_.adaptive_dropout_param().num_output();
		bias_term_ = this->layer_param_.adaptive_dropout_param().bias_term();
		//dropout layers must have same dim of output as input
		//index axis is not included
		N_ = num_output;
		hidden_act_type_ = this->layer_param_.adaptive_dropout_param().neuron_act_type();
		prob_act_type_ = this->layer_param_.adaptive_dropout_param().prob_act_type();
		const int axis = bottom[0]->CanonicalAxisIndex(
			this->layer_param_.adaptive_dropout_param().axis());
		// Dimensions starting from "axis" are "flattened" into a single
		// length K_ vector. For example, if bottom[0]'s shape is (N, C, H, W),
		// and axis == 1, N inner products with dimension CHW are performed.
		//given start axis, return all the count of elements from start axis to total axis.
		K_ = bottom[0]->count(axis);
		// Check if we need to set up the weights
		if (this->blobs_.size() > 0) {
			LOG(INFO) << "Skipping parameter initialization";
		}
		//here the binary belief network shares weights and biases with forward network
		//only use alpha and beta to do affine transformation
		else {
			if (bias_term_) {
				this->blobs_.resize(2);
			}
			else {
				this->blobs_.resize(1);
			}
			// Intialize the weight
			vector<int> weight_shape(2);
			weight_shape[0] = N_;
			weight_shape[1] = K_;
			//reset is a component function of shared pointer, just used to set pointer value
			this->blobs_[0].reset(new Blob<Dtype>(weight_shape));
			// fill the weights
			shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
				this->layer_param_.adaptive_dropout_param().weight_filler()));
			weight_filler->Fill(this->blobs_[0].get());
			// If necessary, intiialize and fill the bias term
			if (bias_term_) {
				vector<int> bias_shape(1, N_);
				this->blobs_[1].reset(new Blob<Dtype>(bias_shape));
				shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
					this->layer_param_.adaptive_dropout_param().bias_filler()));
				bias_filler->Fill(this->blobs_[1].get());
			}
		}  // parameter initialization
		this->param_propagate_down_.resize(this->blobs_.size(), true);
	}

	template <typename Dtype>
	void AdaptiveDropoutLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		// Figure out the dimensions
		const int axis = bottom[0]->CanonicalAxisIndex(
			this->layer_param_.adaptive_dropout_param().axis());
		const int new_K = bottom[0]->count(axis);
		CHECK_EQ(K_, new_K)
			<< "Input size incompatible with inner product parameters.";
		// The first "axis" dimensions are independent inner products; the total
		// number of these is M_, the product over these dimensions.
		M_ = bottom[0]->count(0, axis);
		// The top shape will be the bottom shape with the flattened axes dropped,
		// and replaced by a single axis with dimension num_output (N_).
		vector<int> top_shape = bottom[0]->shape();
		top_shape.resize(axis + 1);
		top_shape[axis] = N_;
		top[0]->Reshape(top_shape);
		//set up cache for the probability vector
		this->prob_vec_.Reshape(top_shape);
		//set up cache for the random vector
		this->rand_vec_.Reshape(top_shape);
		//set up cache for the unact_hidden layer
		this->unact_hidden_.Reshape(top_shape);
		vector<int> weight_shape(2);
		weight_shape[0] = N_;
		weight_shape[1] = K_;
		// Set up the bias multiplier
		if (bias_term_) {
			vector<int> bias_shape(1, M_);
			bias_multiplier_.Reshape(bias_shape);
			caffe_set(M_, Dtype(1), bias_multiplier_.mutable_cpu_data());
		}
	}

	template <typename Dtype>
	inline void common_mul_cpu(const int n, const Dtype* a, const unsigned int* b, Dtype* y){
		for (int i = 0; i < n; i++){
			y[i] = a[i] * Dtype(b[i]);
		}
	}

	template <typename Dtype>
	void AdaptiveDropoutLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		//just one stream of input, so bottom[0] is all the input data
		const Dtype* bottom_data = bottom[0]->cpu_data();
		Dtype* top_data = top[0]->mutable_cpu_data();
		const Dtype* weight_data = this->blobs_[0]->cpu_data();
		//cpu_data(): const Dtype*, can not be changed
		//mutable_cpu_data(): Dtype*, can be set by other values
		Dtype* prob_vec_data = this->prob_vec_.mutable_cpu_data();
		//get probabilities by prob_weight
		caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, N_, K_, (Dtype)1.,
			bottom_data, weight_data, (Dtype)0., prob_vec_data);
		if (bias_term_) {
			//add bias
			caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, 1, (Dtype)1.,
				bias_multiplier_.cpu_data(),
				this->blobs_[1]->cpu_data(), (Dtype)1., prob_vec_data);
		}
		//compute prob_weight_data from weight_data
		//prob_act = f(alpha*(pi * bottom + bias) + beta)
		for (int i = 0; i < this->prob_vec_.count(); i++){
			prob_vec_data[i] = alpha_ * prob_vec_data[i] + beta_;
		}
		//activation for probability
		activate_cpu<Dtype>(prob_vec_.count(), this->prob_vec_.cpu_data(), prob_vec_data, prob_act_type_);

		//weight
		caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, N_, K_, (Dtype)1.,
			bottom_data, weight_data, (Dtype)0., unact_hidden_.mutable_cpu_data());

		const int count_top = top[0]->count();
		//bias
		if (bias_term_) {
			caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, 1, (Dtype)1.,
				bias_multiplier_.cpu_data(),
				this->blobs_[1]->cpu_data(), (Dtype)1., unact_hidden_.mutable_cpu_data());
		}
		//activation for hidden units
		const Dtype* unact_data = unact_hidden_.cpu_data();
		activate_cpu<Dtype>(unact_hidden_.count(), unact_data, top[0]->mutable_cpu_data(), hidden_act_type_);
		//dropout
		if (this->phase_ == TRAIN){
			DCHECK(prob_vec_.count() == rand_vec_.count());
			unsigned int* rand_vec_data = this->rand_vec_.mutable_cpu_data();
			//p[i]=P(r[i]=1)
			caffe_rng_bernoulli<Dtype>(prob_vec_.count(), prob_vec_data, rand_vec_data);
			common_mul_cpu<Dtype>(count_top, top_data, rand_vec_data, top_data);
		}
		else{
			caffe_mul<Dtype>(count_top, top_data, prob_vec_data, top_data);
		}
	}

	template<typename Dtype>
	inline void SigmoidBackward_cpu(const int n, const Dtype* in_diff,
		const Dtype* unact_data, Dtype* out_diff){
		for (int i = 0; i < n; i++){
			const Dtype sigmoid_x = 1. / (1. + exp(-unact_data[i]));
			out_diff[i] = in_diff[i] * sigmoid_x * (1 - sigmoid_x);
		}
	}

	template <typename Dtype>
	inline void ReLUBackward_cpu(const int n, const Dtype* in_diff,
		const Dtype* in_data, Dtype* out_diff){
		for (int i = 0; i<n; i++){
			out_diff[i] = in_diff[i] * (in_data[i] > 0);
		}
	}

	template <typename Dtype>
	inline void ActBackward_cpu(const int n, const Dtype* in_diff,
		const Dtype* in_data, Dtype* out_diff, AdaptiveDropoutParameter_ActType act_type){
		switch (act_type)
		{
		case caffe::AdaptiveDropoutParameter_ActType_RELU:
			ReLUBackward_cpu<Dtype>(n, in_diff, in_data, out_diff);
			break;
		case caffe::AdaptiveDropoutParameter_ActType_SIGMOID:
			SigmoidBackward_cpu<Dtype>(n, in_diff, in_data, out_diff);
			break;
		default:
			LOG(FATAL) << "unknown act function type.";
			break;
		}
	}

	template <typename Dtype>
	void AdaptiveDropoutLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down,
		const vector<Blob<Dtype>*>& bottom) {
		Dtype* top_diff = top[0]->mutable_cpu_diff();
		const int count_top = top[0]->count();
		//backward through dropout, put difference in prob_vec_.diff
		const unsigned int* rand_vec_data = this->rand_vec_.mutable_cpu_data();
		//top_diff = top_diff * rand_vec_data
		common_mul_cpu<Dtype>(count_top, top_diff, rand_vec_data, prob_vec_.mutable_cpu_diff());
		//backward through non-linear activation
		const Dtype* in_data = unact_hidden_.cpu_data();
		ActBackward_cpu<Dtype>(top[0]->count(), prob_vec_.cpu_diff(), in_data,
			unact_hidden_.mutable_cpu_diff(), hidden_act_type_);

		//backward through inner product part
		const Dtype* unact_diff = unact_hidden_.cpu_diff();
		if (this->param_propagate_down_[0]) {
			const Dtype* bottom_data = bottom[0]->cpu_data();
			// Gradient with respect to weight
			caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, N_, K_, M_, (Dtype)1.,
				unact_diff, bottom_data, (Dtype)1., this->blobs_[0]->mutable_cpu_diff());
		}
		if (bias_term_ && this->param_propagate_down_[1]) {
			const Dtype* top_diff = top[0]->cpu_diff();
			// Gradient with respect to bias
			// [N_, M_] x [M_, 1] = N_ x 1
			caffe_cpu_gemv<Dtype>(CblasTrans, M_, N_, (Dtype)1., unact_diff,
				bias_multiplier_.cpu_data(), (Dtype)1.,
				this->blobs_[1]->mutable_cpu_diff());
		}
		if (propagate_down[0]) {
			const Dtype* top_diff = top[0]->cpu_diff();
			// Gradient with respect to bottom data
			caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, K_, N_, (Dtype)1.,
				unact_diff, this->blobs_[0]->cpu_data(), (Dtype)0.,
				bottom[0]->mutable_cpu_diff());
		}
	}

#ifdef CPU_ONLY
	STUB_GPU(AdaptiveDropoutLayer);
#endif

	INSTANTIATE_CLASS(AdaptiveDropoutLayer);
	REGISTER_LAYER_CLASS(AdaptiveDropout);
}