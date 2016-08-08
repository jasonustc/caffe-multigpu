#ifndef CAFFE_VIDEO_UNROLL_LAYER_HPP_
#define CAFFE_VIDEO_UNROLL_LAYER_HPP_

#include "boost/scoped_ptr.hpp"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/filler.hpp"
namespace caffe{
	/**
	* @brief Like reshape_layer, unroll video to frames, and generate continuing indicators used by RCS layers
	*/
	template <typename Dtype>
	class VideoUnrollLayer : public Layer<Dtype> {
	public:
		explicit VideoUnrollLayer(const LayerParameter& param)
			: Layer<Dtype>(param){}
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);

		virtual inline const char* type() const { return "VideoUnroll"; }
		virtual inline int ExactNumBottomBlobs() const { return 1; }
		virtual inline int MinNumTopBlobs() const { return 2; }
		virtual inline int MaxBottomBlobs() const { return 3; }

	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom, 
			const vector<Blob<Dtype>*>& top){}
		virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom, 
			const vector<Blob<Dtype>*>& top){}

		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top, 
			const vector<bool>& propagate_down, 
			const vector<Blob<Dtype>*>& bottom){
			NOT_IMPLEMENTED;
		}
		virtual void Backward_gpu(const vector<Blob<Dtype>*>&  top, 
			const vector<bool>& propagate_down, 
			const vector<Blob<Dtype>*>& bottom){
			NOT_IMPLEMENTED;
		}
	};
}
#endif
