#ifndef CAFFE_ATTENTION_LAYER_
#define CAFFE_ATTENTION_LAYER_

#include "caffe/blob.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/layer.hpp"

namespace caffe{
	/*
	 * @brief: attention is about to generate weighted output
	 * Y := W op X
	 * here the op specifies different type of weighted operations
	 * like broadcasted weights, filter banks et al.
	 */
	template <typename Dtype>
	class AttentionLayer : public Layer<Dtype>{
	public:
		explicit AttentionLayer(const LayerParameter& param)
			: Layer<Dtype>(param){}
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual inline const char* type() const { return "Attention"; }

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
		virtual inline int MinBottomBlobs() const { return 2; }
	};
} // namespace caffe

#endif