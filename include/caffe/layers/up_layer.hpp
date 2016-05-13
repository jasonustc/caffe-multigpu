#ifndef CAFFE_UP_LAYER_HPP_
#define CAFFE_UP_LAYER_UPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/transformation.hpp"

namespace caffe{
	// AJ: Upsampling layer, that takes in one bottom and returns #
	// transformations+1 many tops
	template <typename Dtype> class UpsamplingLayer : public Layer<Dtype> {
	public:
		explicit UpsamplingLayer(const LayerParameter &param) : Layer<Dtype>(param) {}
		// Not adhereing with the LayerSetup/Reshape setup for this layer because
		// Reshape will recompute the transformation indices and it's waste of
		// computation. Keep reshape empty & do everything in LayerSetUp.
		virtual void LayerSetUp(const vector<Blob<Dtype> *> &bottom,
			const vector<Blob<Dtype> *>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top) {};

		virtual inline const char* type() const {
			return "Up";
		}
		virtual inline int MinBottomBlobs() const { return 1; }
		virtual inline int MinTopBlobs() const { return 1; }

	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype> *> &bottom,
			const vector<Blob<Dtype> *> &top);
		virtual void Forward_gpu(const vector<Blob<Dtype> *> &bottom,
			const vector<Blob<Dtype> *>& top);
		virtual void Backward_cpu(const vector<Blob<Dtype> *> &top,
			const vector<bool>& propagate_down,
			const vector<Blob<Dtype>*>& bottom);
		virtual void Backward_gpu(const vector<Blob<Dtype> *> &top,
			const vector<bool>& propagate_down,
			const vector<Blob<Dtype> *>& bottom);

		int NUM_T_;
		int HEIGHT_;    // of input
		int WIDTH_;     // of input
		Border BORDER_; // of transformation
		Interp INTERP_; // of transformation
		// 3x3 rotation matrix buffer, row ordater:
		float tmat_[9];
		// Indices for image transformation
		// We use blob's data to be fwd and diff to be backward indices
		vector<shared_ptr<Blob<float> > > coord_idx_;

		// bc TIConvolutionLayer has to call F/Bward_g/cpu functions
		template <typename D> friend class TIConvolutionLayer;
	};
}
#endif