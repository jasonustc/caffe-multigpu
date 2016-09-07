#ifndef CAFFE_SAMPLING_LAYER_HPP_
#define CAFFE_SAMPLING_LAYER_HPP_
#include <vector>
#include <utility>

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/math_functions.hpp"

namespace caffe{
	/*
	* since we already have exp layer and innerproduct layer in caffe
	* we directly use them to calculate u_t = W_u * input + b_u and \sigma_t = exp(W_\sigma * input + b_\sigma)
	* so here we take vector u and vector sigma as input, and use them to sample gaussian values
	*/
	template <typename Dtype>
	class SamplingLayer : public Layer<Dtype> {
	public:
		explicit SamplingLayer(const LayerParameter& param) : Layer<Dtype>(param){}
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual inline const char* type() const { return "SamplingLayer"; }

	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down,
			const vector<Blob<Dtype>*>& bottom);
		virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down,
			const vector<Blob<Dtype>*>& bottom);

		virtual inline int MaxNumBottomBlobs() const { return 2; }
		virtual inline int MinNumBottomBlobs() const { return 1; }
		virtual inline int ExactNumTopBlobs() const { return 1; }

		//the sampling value of standard gaussian distribution
		//needed in backpropagation
		Blob<Dtype> gaussian_value_;
		// currently, backward is only allowed in GAUSSIAN sampling
		bool is_gaussian_;
		SamplingParameter_SampleType sample_type_;
	};

} // namespace caffe

#endif
