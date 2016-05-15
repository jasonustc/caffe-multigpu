#ifndef CAFFE_DOWNPOOL_LAYER_HPP_
#define CAFFE_DOWNPOOL_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"


namespace caffe{
	// AJ: DownPooling layer, that takes in multiple bottoms and returns 1 top
	// reinterpolate all the bottom into a canonical shape (shape of bottom[0])
	// then computes max/ave pooling across all bottoms (transformations)
	template <typename Dtype> class DownPoolingLayer : public Layer<Dtype> {
	public:
		explicit DownPoolingLayer(const LayerParameter &param)
			: Layer<Dtype>(param), switch_idx_(0, 0, 0, false) {}
		// Same as Upsample, Reshape does nothing, everythin is done in LayerSetUp.
		virtual void LayerSetUp(const vector<Blob<Dtype> *> &bottom,
			const vector<Blob<Dtype> *> &top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*> &top) {};

		virtual inline const char* type() const {
			return "Downpooling";
		}
		virtual inline int MinBottomBlobs() const { return 1; }
		virtual inline int ExactNumTopBlobs() const { return 1; }

		virtual void Report(const std::string &name);

		const Blob<float> &max_switch() { return switch_idx_; }
		const vector<shared_ptr<Blob<float> > > &coord_idx() { return coord_idx_; }

		inline const vector<float> &trans_counter() { return trans_counter_; }

	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype> *> &bottom,
			const vector<Blob<Dtype> *> &top);
		virtual void Forward_gpu(const vector<Blob<Dtype> *> &bottom,
			const vector<Blob<Dtype> *> &top);
		virtual void Backward_cpu(const vector<Blob<Dtype> *> &top,
			const vector<bool>& propagate_down,
			const vector<Blob<Dtype> *>& bottom);
		virtual void Backward_gpu(const vector<Blob<Dtype> *> &top,
			const vector<bool>& propagate_down,
			const vector<Blob<Dtype> *> &bottom);
		void UpdateCounter(const int *curr_counter, const int &top_count);

		int NUM_T_;
		int HEIGHT_; // of output
		int WIDTH_;  // of output
		int NUM_OUTPUT_;
		int CHANNEL_OUTPUT_;
		Border BORDER_; // of transformation
		Interp INTERP_; // of transformation

		// 3x3 rotation matrix buffer:
		float tmat_[9];
		// Indices for image transformation
		// We use blob's data to be fwd and diff to be backward indices
		vector<shared_ptr<Blob<float> > > coord_idx_;
		// Switches for max-pooling over transformations
		Blob<float> switch_idx_;
		// TODO(AJ): Since top sizes are all going to be the same we don't need to
		// have this top_buffer_ as a vector
		vector<shared_ptr<Blob<Dtype> > > top_buffer_;

		// For debugging/printing how often each transformation was used
		vector<float> trans_counter_;

		// bc TIConvolutionLayer has to call F/Bward_g/cpu functions
		template <typename D> friend class TIConvolutionLayer;
	};
}
#endif
