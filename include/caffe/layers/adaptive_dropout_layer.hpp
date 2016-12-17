/*
 * 2016/12/17(xu): NOTE, this version of code is kind of redundant, need 
 * to refine it.
 */
#include <vector>
#include <string>

#include "caffe/blob.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/layer.hpp"
#include "caffe/filler.hpp"

namespace caffe{
	//TODO: use independent weight for adaptive dropout probability
	template <typename Dtype>
	class AdaptiveDropoutLayer : public Layer<Dtype> {
	public:
		explicit AdaptiveDropoutLayer(const LayerParameter& param)
			: Layer<Dtype>(param) {}
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual inline const char* type() const { return "AdaptiveDropout"; }
		virtual inline int ExactNumBottomBlobs() const { return 1; }
		virtual inline int ExactNumTopBlobs() const { return 1; }

	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
		virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

		//count of all data
		int M_;
		int K_;
		//number of output
		int N_;
		bool bias_term_;
		Blob<Dtype> bias_multiplier_;
		//adaptive dropout probability
		Blob<Dtype> prob_vec_;
		//the raw hidden layer value before activate
		Blob<Dtype> unact_hidden_;
		//random generated number
		Blob<unsigned int> rand_vec_;
		//affine parameters of the relationship between action 
		//weight and dropout weight
		Dtype alpha_;
		Dtype beta_;
		//activation type for hidden layers
		AdaptiveDropoutParameter_ActType hidden_act_type_;
		//activation type for probability
		AdaptiveDropoutParameter_ActType prob_act_type_;
	};
}