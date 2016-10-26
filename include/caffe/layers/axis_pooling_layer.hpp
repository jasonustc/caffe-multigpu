#ifndef CAFFE_AXIS_POOLING_LAYER_HPP_
#define CAFFE_AXIS_POOLING_LAYER_HPP_

#include <vector>


#include "caffe/blob.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/layer.hpp"

namespace caffe {
	/**
	* @brief Takes one blob, pool it on certain axis
	*/
	template <typename Dtype>
	class AxisPoolingLayer : public Layer<Dtype> {
	public:
		explicit AxisPoolingLayer(const LayerParameter& param)
			: Layer<Dtype>(param) {}
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual inline const char* type() const { return "AxisPooling"; }
		virtual inline int ExactNumBottomBlobs() const { return 1; }
		virtual inline int ExactNumTopBlobs() const { return 1; }

	protected:
		/**
		* @param bottom input Blob vector (length 1)
		* @param top output Blob vector (length 1)
		*/
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);

		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
		virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

		int pool_axis_;
		int num_pools_;
		int pool_input_size_;
		Blob<int> max_idx_;
	};
} // namespace caffe

#endif