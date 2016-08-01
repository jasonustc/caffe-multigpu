#ifndef CAFFE_GEN_REC_LOSS_LAYER_HPP_
#define CAFFE_GEN_REC_LOSS_LAYER_HPP_

#include <vector>
#include <utility>

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/loss_layer.hpp"

#define PI 3.1415926

namespace caffe{
	/*
	* Please refer to paper DRAW: A Recurrent Neural Network For Image Generation for
	* more information
	* L^{x_t} = - logD(x_t|c_t) in original paper
	* L^{x_t} = 1/2 * N * log2\pi -1/2 * (x-\mu)^2 / \sigma^2 + \sum log\sigma^2 here
	* May need to do some adjust from video frames input
	* @param Bottom input blob vector(length 3): \mu, \sigma and x_raw
	*/
	template <typename Dtype>
	class GenRecLossLayer : public LossLayer<Dtype>{
	public:
		explicit GenRecLossLayer(const LayerParameter& param) : LossLayer<Dtype>(param){}
		virtual inline const char* type()const { return "Generative Reconstruction Loss"; }
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
		virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
			const vector<Blob<Dtype>*>& bottom);
		virtual void Backward_gpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
			const vector<Blob<Dtype>*>& bottom);

		virtual inline int ExactNumBottomBlobs() const { return 3; }
		//can not propagate error to data x
		virtual inline bool AllowForceBackward(const int index){ return index != 2; }

		Blob<Dtype> sum_multiplier_;
		//put buffer into mu_sigma_buffer_ data and mu_sigma_buffer_ diff
		Blob<Dtype> mu_sigma_buffer_;
		int num_feats_;
	};
} // namespace caffe
#endif