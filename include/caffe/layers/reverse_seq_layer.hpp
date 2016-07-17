/********************************************************************************
** Copyright(c) 2016 USTC All Rights Reserved.
** auth£º Xu Shen
** mail£º shenxuustc@gmail.com
** date£º 2016/07/01
** desc£º reverse sequence layer
*********************************************************************************/
#ifndef CAFFE_REVERSE_SEQUENCE_LAYER_HPP_
#define CAFFE_REVERSE_SEQUENCE_LAYER_HPP_

#include <string>
#include <utility>
#include <vector>

#include "boost/scoped_ptr.hpp"
#include "caffe/blob.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/layer.hpp"
#include "caffe/layers/slice_layer.hpp"
#include "caffe/layers/concat_layer.hpp"

namespace caffe{
	/**
	 * @brief reverse the order of input sequence
	 * @input: X_, cont_
	 * @output: X_reversed_
	 **/
	template <typename Dtype>
	class ReverseSeqLayer : public Layer<Dtype>{
	public: 
		explicit ReverseSeqLayer(const LayerParameter& param)
			: Layer<Dtype>(param){}

		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual inline int ExactNumBottomBlobs() const { return 2; }
		virtual inline int ExactNumTopBlobs() const { return 1; }
		virtual inline bool AllowForceBackward(const int bottom_index) const{
			// Can't propagate to sequence continuation indicators.
			return bottom_index != 1;
		}

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

		void GetIndex(const vector<Blob<Dtype>*>& bottom);

		Blob<int> index_;
	};
}

#endif //CAFFE_REVERSE_SEQ_LAYER_HPP_
