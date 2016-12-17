#include <vector>

#include "caffe/layers/dropconnect_layer.hpp"

namespace caffe{
	template <typename Dtype>
	void DropConnectLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		threshold_ = this->layer_param_.dropconnect_param().dropout_ratio();
		DCHECK(threshold_ > 0.);
		DCHECK(threshold_ < 1.);
		scale_ = 1. / (1. - threshold_);
		uint_thres_ = static_cast<unsigned int>(UINT_MAX * threshold_);
		const int num_output = this->layer_param_.dropconnect_param().num_output();
		bias_term_ = this->layer_param_.dropconnect_param().bias_term();
		N_ = num_output;
		const int axis = bottom[0]->CanonicalAxisIndex(
			this->layer_param_.dropconnect_param().axis());
		// Dimensions starting from "axis" are "flattened" into a single
		// length K_ vector. For example, if bottom[0]'s shape is (N, C, H, W),
		// and axis == 1, N inner products with dimension CHW are performed.
		//given start axis, return all the count of elements from start axis to total axis.
		K_ = bottom[0]->count(axis);
		// Check if we need to set up the weights
		if (this->blobs_.size() > 0) {
			LOG(INFO) << "Skipping parameter initialization";
		}
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
				this->layer_param_.dropconnect_param().weight_filler()));
			weight_filler->Fill(this->blobs_[0].get());
			// If necessary, intiialize and fill the bias term
			if (bias_term_) {
				vector<int> bias_shape(1, N_);
				this->blobs_[1].reset(new Blob<Dtype>(bias_shape));
				shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
					this->layer_param_.dropconnect_param().bias_filler()));
				bias_filler->Fill(this->blobs_[1].get());
			}
		}  // parameter initialization
		this->param_propagate_down_.resize(this->blobs_.size(), true);
	}

	template <typename Dtype>
	void DropConnectLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		// Figure out the dimensions
		const int axis = bottom[0]->CanonicalAxisIndex(
			this->layer_param_.dropconnect_param().axis());
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
		vector<int> weight_shape(2);
		weight_shape[0] = N_;
		weight_shape[1] = K_;
		top[0]->Reshape(top_shape);
		// Set up the cache for random weight multiplier generation
		this->weight_multiplier_.Reshape(this->blobs_[0]->shape());
		//Initialize the weight after dropout
		//one sample one channel corresponding to one time drop connect
		this->dropped_weight_.Reshape(this->blobs_[0]->shape());
		// Set up the bias multiplier
		if (bias_term_) {
			vector<int> bias_shape(1, M_);
			bias_multiplier_.Reshape(bias_shape);
			caffe_set(M_, Dtype(1), bias_multiplier_.mutable_cpu_data());
		}
	}

	template <typename Dtype>
	void DropConnectLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		const Dtype* bottom_data = bottom[0]->cpu_data();
		Dtype* top_data = top[0]->mutable_cpu_data();
		const Dtype* weight = this->blobs_[0]->cpu_data();
		//cpu_data(): const Dtype*, can not be changed
		//mutable_cpu_data(): Dtype*, can be set by other values
		unsigned int* weight_multiplier = this->weight_multiplier_.mutable_cpu_data();
		Dtype* dropped_weight = this->dropped_weight_.mutable_cpu_data();
		const int count = this->blobs_[0]->count();
		//weight: N_ x K_
		if (this->phase_ == TRAIN){
			//create random numbers
			caffe_rng_bernoulli(count, 1. - threshold_, weight_multiplier);
			for (int i = 0; i < count; i++){
				dropped_weight[i] = weight_multiplier[i] * weight[i] * scale_;
			}
		}
		else{
			caffe_copy(count, weight, dropped_weight);
		}
		//output = W^T * Input
		caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, N_, K_, (Dtype)1.,
			bottom_data, dropped_weight, (Dtype)0., top_data);
		if (bias_term_) {
			//top_data = 1* top_data + bias_multiplier * blobs_[1]
			caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, 1, (Dtype)1.,
				bias_multiplier_.cpu_data(),
				this->blobs_[1]->cpu_data(), (Dtype)1., top_data);
		}
	}

	template <typename Dtype>
	void DropConnectLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down,
		const vector<Blob<Dtype>*>& bottom) {
		if (this->param_propagate_down_[0]) {
			const Dtype* top_diff = top[0]->cpu_diff();
			const Dtype* bottom_data = bottom[0]->cpu_data();
			// Gradient with respect to weight
			Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();
			const unsigned int* weight_multiplier = weight_multiplier_.cpu_data();
			const int count_weight = this->blobs_[0]->count();

			caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, N_, K_, M_, (Dtype)1.,
				top_diff, bottom_data, (Dtype)0., this->blobs_[0]->mutable_cpu_diff());
			if (this->phase_ == TRAIN){
				for (int i = 0; i < count_weight; i++){
					weight_diff[i] = weight_diff[i] * weight_multiplier[i] * scale_;
				}
			}
		}
		if (bias_term_ && this->param_propagate_down_[1]) {
			const Dtype* top_diff = top[0]->cpu_diff();
			// Gradient with respect to bias
			caffe_cpu_gemv<Dtype>(CblasTrans, M_, N_, (Dtype)1., top_diff,
				bias_multiplier_.cpu_data(), (Dtype)0.,
				this->blobs_[1]->mutable_cpu_diff());
		}
		if (propagate_down[0]) {
			const Dtype* top_diff = top[0]->cpu_diff();
			// Gradient with respect to bottom data
			caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, K_, N_, (Dtype)1.,
				top_diff, this->dropped_weight_.cpu_data(), (Dtype)0.,
				bottom[0]->mutable_cpu_diff());
		}
	}

#ifdef CPU_ONLY
	STUB_GPU(DropConnectLayer);
#endif

	INSTANTIATE_CLASS(DropConnectLayer);
	REGISTER_LAYER_CLASS(DropConnect);
}