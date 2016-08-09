#include <vector>
#include <utility>

#include "caffe/layers/rbm_layer.hpp"

namespace caffe{
	template <typename Dtype>
	inline Dtype sigmoid(Dtype x){
		return 1. / (1. + exp(-x));
	}

	template <typename Dtype>
	void RBMLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top){
		//TODO: implement CD-k not only CD-1 in current version
		num_iteration_ = this->layer_param_.rbm_param().num_iteration();
		CHECK_GE(num_iteration_, 1) << "iteration times should be at least 1.";
		const int num_output = this->layer_param_.rbm_param().num_output();
		bias_term_ = this->layer_param_.rbm_param().has_bias_filler();
		N_ = num_output;
		//when we use axis from proto file, we'd better call this CanonicalAxisIndex function
		//to make sure it indexes the right axis
		const int axis = bottom[0]->CanonicalAxisIndex(this->layer_param_.rbm_param().axis());
		// Dimensions starting from "axis" are "flattened" into a single
		// length K_ vector. For example, if bottom[0]'s shape is (N, C, H, W),
		// and axis == 1, N inner products with dimension CHW are performed.
		K_ = bottom[0]->count(axis);
		if (this->blobs_.size() > 0){
			LOG(INFO) << "Skipping parameter initialization";
		}
		else{
			//in RBM layer, we have h_bias and v_bias seperately
			if (bias_term_){
				this->blobs_.resize(3);
			}
			else{
				this->blobs_.resize(1);
			}

			//Initialize the weight
			vector<int> weight_shape(2);
			weight_shape[0] = N_;
			weight_shape[1] = K_;
			this->blobs_[0].reset(new Blob<Dtype>(weight_shape));
			//fill the weights
			shared_ptr<Filler<Dtype>> weight_filler(GetFiller<Dtype>(
				this->layer_param_.rbm_param().weight_filler()));
			weight_filler->Fill(this->blobs_[0].get());
			//If necessary, initialize and fill the bias term
			//bias has the same dim with the corresponding data
			if (bias_term_){
				vector<int> h_bias_shape(1, N_);
				vector<int> v_bias_shape(1, K_);
				this->blobs_[1].reset(new Blob<Dtype>(h_bias_shape));
				this->blobs_[2].reset(new Blob<Dtype>(v_bias_shape));
				shared_ptr<Filler<Dtype>> bias_filler(GetFiller<Dtype>(
					this->layer_param_.rbm_param().bias_filler()));
				bias_filler->Fill(this->blobs_[1].get());
				bias_filler->Fill(this->blobs_[2].get());
			}
		} // parameter initialization
		this->param_propagate_down_.resize(this->blobs_.size(), true);
		//TODO: implement different types of sampling, not just binary 
		//sampling in current version
		//actually, in RBM, bernoulli sampling is enough
		sample_type_ = this->layer_param_.rbm_param().sample_type();
	}

	template <typename Dtype>
	void RBMLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top){
		//Figure out the dimensions
		const int axis = bottom[0]->CanonicalAxisIndex(
			this->layer_param_.rbm_param().axis());
		const int new_K = bottom[0]->count(axis);
		DCHECK_EQ(K_, new_K)
			<< "Input size incompatible with inner product parameters.";
		//The first "axis" dimensions are independent inner products; the total
		//number of these is M_, the product over these dimensions.
		M_ = bottom[0]->count(0, axis);
		//The top shape will be the bottom shape with the flattened axis dropped,
		// and replaced by a single axis with dimension num_out (N_)
		vector<int> top_shape = bottom[0]->shape();
		top_shape.resize(axis + 1);
		top_shape[axis] = N_;
		top[0]->Reshape(top_shape);
		pos_h_.ReshapeLike(*top[0]);
		//only share data, not share diff
		pos_h_.ShareData(*top[0]);
		pos_v_.ReshapeLike(*bottom[0]);
		pos_v_.ShareData(*bottom[0]);
		//only share data, not share diff
		neg_h_.Reshape(top_shape);
		positive_state_h_.Reshape(top_shape);
		negative_state_v_.ReshapeLike(*bottom[0]);
		neg_v_.ReshapeLike(*bottom[0]);

		//buffer for weight diff
		weight_diff_buf_.ReshapeLike(*this->blobs_[0]);

		if (bias_term_){
			vector<int> bias_shape(1, M_);
			bias_multiplier_.Reshape(bias_shape);
			caffe_set(M_, Dtype(1), bias_multiplier_.mutable_cpu_data());
		}
		//output reconstruction loss
		//TODO: output other types of loss, like negative log likelihood
		if (top.size() > 1){
			vector<int> loss_shape(0);
			top[1]->Reshape(loss_shape);
		}
	}

	//TODO: currently only one step of CD is implemented,
	//CD-k is to be updated
	template <typename Dtype>
	void RBMLayer<Dtype>::Gibbs_vhvh_cpu(){
		const Dtype* weight_data = this->blobs_[0]->cpu_data();
		const Dtype* h_bias_data = this->blobs_[1]->cpu_data();
		const Dtype* v_bias_data = this->blobs_[2]->cpu_data();
		Dtype* pos_h_data = pos_h_.mutable_cpu_data();
		Dtype* neg_h_data = neg_h_.mutable_cpu_data();
		Dtype* positive_state_h_data = positive_state_h_.mutable_cpu_data();
		Dtype* negative_state_v_data = negative_state_v_.mutable_cpu_data();
		const Dtype* pos_v_data = pos_v_.cpu_data();
		Dtype* neg_v_data = neg_v_.mutable_cpu_data();
		const int count_h = pos_h_.count();
		const int count_v = neg_v_.count();
		//prop up
		//h: M x N  v: M x K w: N x K
		caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, N_, K_, (Dtype)1.,
			pos_v_data, weight_data, (Dtype)0, pos_h_data);
		if (bias_term_){
			caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, 1, (Dtype)1.,
				bias_multiplier_.cpu_data(), h_bias_data, (Dtype)1., pos_h_data);
		}
		//sigmoid activation
		for (int i = 0; i < count_h; i++){
			pos_h_data[i] = sigmoid(pos_h_data[i]);
		}
		//sampling
		caffe_rng_bernoulli<Dtype>(count_h, pos_h_data, positive_state_h_data);

		//prop down
		//h: M x N  v: M x K w: N x K
		caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, K_, N_, (Dtype)1.,
			positive_state_h_data, weight_data, (Dtype)0., neg_v_data);
		if (bias_term_){
			caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, K_, 1, (Dtype)1.,
				bias_multiplier_.cpu_data(), v_bias_data, (Dtype)1., neg_v_data);
		}
		//sigmoid activation
		for (int i = 0; i < count_v; i++){
			neg_v_data[i] = sigmoid(neg_v_data[i]);
		}
		//sampling 
		caffe_rng_bernoulli<Dtype>(count_v, neg_v_data, negative_state_v_data);

		//prop up again
		caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, N_, K_, (Dtype)1.,
			negative_state_v_data, weight_data, (Dtype)0, neg_h_data);
		if (bias_term_){
			caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, 1, (Dtype)1.,
				bias_multiplier_.cpu_data(), h_bias_data, (Dtype)1., neg_h_data);
		}

		//sigmoid activation
		for (int i = 0; i < count_h; i++){
			neg_h_data[i] = sigmoid(neg_h_data[i]);
		}
	}

	template <typename Dtype>
	void RBMLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top){
		Gibbs_vhvh_cpu();
		const Dtype* bottom_data = bottom[0]->cpu_data();
		if (top.size() > 1){
			const int count = bottom[0]->count();
			//use neg_v diff for buffer of reconstruction error data
			caffe_sub<Dtype>(count, bottom_data, neg_v_.cpu_data(), neg_v_.mutable_cpu_diff());
			Dtype loss = caffe_cpu_dot<Dtype>(count, neg_v_.cpu_diff(), neg_v_.cpu_diff());
			top[1]->mutable_cpu_data()[0] = loss / bottom[0]->num();
		}
	}

	template <typename Dtype>
	void RBMLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down,
		const vector<Blob<Dtype>*>& bottom){
		//put positive data into buf data
		Dtype* pos_ass_data = weight_diff_buf_.mutable_cpu_data();
		//put negative data into buf diff
		Dtype* neg_ass_data = weight_diff_buf_.mutable_cpu_diff();
		const Dtype* pos_v_data = bottom[0]->cpu_data();
		const Dtype* pos_h_data = pos_h_.cpu_data();
		const Dtype* neg_v_data = neg_v_.cpu_data();
		const Dtype* neg_h_data = neg_h_.cpu_data();
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

		const int count_h = pos_h_.count();
		Dtype* h_bias_diff = this->blobs_[1]->mutable_cpu_diff();
		//Gradient with respect to h_bias
		//\delta c_j = \delta c_j + p_h_j^(0) - p_h_j^(k)
		if (bias_term_ && this->param_propagate_down_[1]){
			//put buffer data in neg_h_.diff()
			//pos_h_ is shared with top[0], be carefully to use it in other place
			caffe_sub<Dtype>(count_h, pos_h_data, neg_h_data, neg_h_.mutable_cpu_diff());
			//average by batch size
			//(M_, N_) here is the raw rows and cols of A
			caffe_cpu_gemv<Dtype>(CblasTrans, M_, N_, scale, neg_h_.cpu_diff(),
				bias_multiplier_.cpu_data(), (Dtype)1., h_bias_diff);
		}

		const int count_v = pos_v_.count();
		Dtype* v_bias_diff = this->blobs_[2]->mutable_cpu_diff();
		//Gradient with respect to v_bias
		//\delta b_j = \delta b_j + v_j^(0) - v_j^(k)
		if (bias_term_ && this->param_propagate_down_[2]){
			//put buffer data in neg_v_.diff()
			//pos_v_ is shared with bottom[0], be carefully to use it in other place
			caffe_sub<Dtype>(count_v, pos_v_data, neg_v_data, neg_v_.mutable_cpu_diff());
			//put intemidiate result into neg_v_ data()
			//average by batch size
			//(M_, K_) here is the raw rows and cols of A
			caffe_cpu_gemv<Dtype>(CblasTrans, M_, K_, scale, neg_v_.cpu_diff(),
				bias_multiplier_.cpu_data(), (Dtype)1., v_bias_diff);
		}

		//Gradient with respect to bottom data
		if (propagate_down[0]){
			caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, K_, N_, (Dtype)1.,
				top_diff, weight_data, (Dtype)0., bottom_diff);
		}
	}

#ifdef CPU_ONLY
	STUB_GPU(RBMLayer);
#endif

	INSTANTIATE_CLASS(RBMLayer);
	REGISTER_LAYER_CLASS(RBM);
} // namespace caffe