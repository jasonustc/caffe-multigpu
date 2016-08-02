#ifndef CAFFE_NOISE_LAYER_HPP_
#define CAFFE_NOISE_LAYER_HPP_
#include <vector>
#include <utility>

#include "caffe/proto/caffe.pb.h"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/blob.hpp"

namespace caffe{
	/**
	* @brief To randomly add noise to the input
	* In auto-encoder, this can train a generative model
	* to learn the distribution of the input rather than
	* just the code.
	*/
	template <typename Dtype>
	class NoiseLayer : public Layer<Dtype> {
	public:
		/*
		*@param distribution type and corresponding parameters
		*/
		explicit NoiseLayer(const LayerParameter& param)
			: Layer<Dtype>(param){}
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual inline const char* type() const { return "Noise"; }

	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
		virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

		virtual inline int ExtactNumBottomBlobs() const { return 1; }
		virtual inline int ExtactNumTopBlobs() const { return 1; }

		//parameter for noise distribution
		Dtype alpha_;
		Dtype beta_;

		//noise type
		NoiseParameter_NoiseType noise_type_;

		//apply type
		NoiseParameter_ApplyType apply_type_;

		//noise value
		Blob<Dtype> noise_;
	};

} // namespace caffe
#endif