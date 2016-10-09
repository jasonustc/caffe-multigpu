#ifndef CAFFE_LATENT_LOSS_LAYER_HPP_
#define CAFFE_LATENT_LOSS_LAYER_HPP_

#include "caffe/blob.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/layer.hpp"
#include "caffe/layers/loss_layer.hpp"

namespace caffe{

	const float PI = 3.1415926;
	/*
	* Please refer to paper DRAW: A Recurrent Neural Network For Image Generation for
	* more information
	* L^z = 1/2(\sum_{t=1}^{T}(\mu_t^2 + \sigma_t^2 - log\sigma_t^2) - T/2
	* @param Bottom input blob vector(length 2): \mu and \sigma
	*/

	template <typename Dtype>
	class LatentLossLayer : public LossLayer<Dtype>{
	public:
		explicit LatentLossLayer(const LayerParameter& param) : LossLayer<Dtype>(param){}
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
		virtual inline const char* type() const { return "LatentLoss"; }

	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
		virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
			const vector<Blob<Dtype>*>& bottom);
		virtual void Backward_gpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
			const vector<Blob<Dtype>*>& bottom);

		virtual inline int ExactNumBottomBlobs()const { return 2; }

		Blob<Dtype> log_square_sigma_;
		Blob<Dtype> sum_multiplier_;
	};

}
#endif