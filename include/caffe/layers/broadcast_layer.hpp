#ifndef CAFFE_BROADCAST_LAYER_HPP_
#define CAFFE_BROADCAST_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {
	/*
	 * @brief broast in given axis of bottom[0] to have the same shape with 
	 * bottom[1], bottom[0] must have the same dimension with bottom[1] before axis
	 * the dimension of axis in bottom[1] must be divided by bottom[0]
	 */
	template <typename Dtype>
	class BroadcastLayer : public Layer<Dtype>{
	public:
		explicit BroadcastLayer(const LayerParameter& param):
			Layer<Dtype>(param){}
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual inline int ExactNumBottomBlobs() const { return 2; }
		virtual inline int ExactNumTopBlobs() const { return 1; }
		virtual inline const char* type() const { return "Broadcast"; }
		virtual inline bool AllowForceBackward(const int bottom_index) const {
			// Can't propagate to reference blob.
			return bottom_index != 1;
		}

	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down,
			const vector<Blob<Dtype>*>& bottom);
		virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down,
			const vector<Blob<Dtype>*>& bottom);

	private:
		int axis_;
	};

} // namespace caffe

#endif