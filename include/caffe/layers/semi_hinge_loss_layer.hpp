#ifndef CAFFE_SEMI_HINGE_LOSS_LAYER_
#define CAFFE_SEMI_HINGE_LOSS_LAYER_
#include<vector>
#include<utility>

#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/layer.hpp"
#include "caffe/layers/loss_layer.hpp"
#include "caffe/blob.hpp"

namespace caffe{

	template <typename Dtype>
	class SemiHingeLossLayer : public LossLayer<Dtype>{
	public:
		explicit SemiHingeLossLayer(const LayerParameter& param)
			: LossLayer<Dtype>(param){}
		virtual inline const char* type() const { return "SemiHingeLoss"; }

	protected:
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
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
		virtual inline int ExactNumBottomBlobs() const { return 4; }
		virtual inline bool AllowForceBackward(const int bottom_index) const{
			// can not propagate to label
			return bottom_index != 2 && bottom_index != 3;
		}

		// the label that indicates missing of ground truth label of input data
		int ignore_label_;
		Dtype sup_bias_;
		Dtype unsup_bias_;
		// weight of unsupervised loss, the weight of supervised loss is assumed to be 1.
		Dtype gamma_;
		Dtype sup_thre_;
		Dtype unsup_thre_;
		int axis_;

		// diff for distance computing
		shared_ptr<Blob<Dtype> > diff_;
		// hinge loss of the distance
		shared_ptr<Blob<Dtype> > dist_;
	};
} // namespace caffe

#endif