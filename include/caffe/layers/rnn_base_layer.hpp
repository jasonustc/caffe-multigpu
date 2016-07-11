
/********************************************************************************
** Copyright(c) 2016 USTC & MSRA All Rights Reserved.
** auth： Zhaofan Qiu
** mail： zhaofanqiu@gmail.com
** date： 2016/3/27
** desc： RNNBase layer
*********************************************************************************/

#ifndef CAFFE_RNN_BASE_LAYER_HPP_
#define CAFFE_RNN_BASE_LAYER_HPP_

#include <string>
#include <utility>
#include <vector>

#include "leveldb/db.h"
#include "boost/scoped_ptr.hpp"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/net.hpp"

#include "caffe/layers/slice_layer.hpp"
#include "caffe/layers/concat_layer.hpp"

#include "caffe/layers/argmax_layer.hpp"
#include "caffe/layers/reshape_layer.hpp"
#include "caffe/layers/embed_layer.hpp"
#include "caffe/layers/inner_product_layer.hpp"

namespace caffe {
	/**
	* @brief Implementation of LSTM
	*/
	template <typename Dtype>
	class RNNBaseLayer : public Layer<Dtype> {
	public:
		explicit RNNBaseLayer(const LayerParameter& param)
			: Layer<Dtype>(param) {}

		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual inline int ExactNumBottomBlobs() const { return 2; }
		virtual inline int MinTopBlobs() const { return 1; }
		virtual inline int MaxTopBlobs() const { return 2; }

		virtual inline bool AllowForceBackward(const int bottom_index) const {
			// Can't propagate to sequence continuation indicators.
			return bottom_index != 1;
		}

		virtual vector<Blob<Dtype>*> RecurrentOutput() = 0;

	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

		virtual void RecurrentForward(const int t) = 0;
		virtual void RecurrentBackward(const int t) = 0;
		virtual void CopyRecurrentOutput() = 0;
		virtual void ShareWeight() = 0;
		virtual int GetHiddenDim() = 0;
		
		int hidden_dim_;
		int T_;
		int X_dim_;
		// slice_x_ layer
		shared_ptr<SliceLayer<Dtype> > slice_x_;
		vector<shared_ptr<Blob<Dtype> > > X_;
		// slice_cont_ layer
		shared_ptr<SliceLayer<Dtype> > slice_cont_;
		vector<shared_ptr<Blob<Dtype> > > CONT_;
		// concat_ht_ layer
		shared_ptr<ConcatLayer<Dtype> > concat_ht_;
		vector<shared_ptr<Blob<Dtype> > > H_;

	};
}  // namespace caffe

#endif  // CAFFE_RNN_BASE_LAYER_HPP_
