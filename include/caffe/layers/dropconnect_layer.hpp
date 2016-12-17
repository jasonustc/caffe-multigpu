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
	/**
	* @brief Also known as a "fully-connected" layer, computes an inner product
	*        with a set of learned weights, and (optionally) adds biases.
	*
	* TODO(dox): thorough documentation for Forward, Backward, and proto params.
	*/
	template <typename Dtype>
	class DropConnectLayer : public Layer<Dtype> {
	public:
		explicit DropConnectLayer(const LayerParameter& param)
			: Layer<Dtype>(param) {}
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual inline const char* type() const { return "DropConnect"; }
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
		//drop connect blob
		Blob<unsigned int> weight_multiplier_;
		Blob<Dtype> dropped_weight_;
		//the probability @f$ p @f$ of dropping any weight
		Dtype threshold_;
		//the scale for undropped inputs at train time @f$ 1 / (1 - p) @f$
		Dtype scale_;
		unsigned int uint_thres_;
	};
}
