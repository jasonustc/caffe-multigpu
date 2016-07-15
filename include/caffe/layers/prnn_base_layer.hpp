/********************************************************************************
** Copyright(c) 2016 USTC & MSRA All Rights Reserved.
** auth£º Xu Shen
** mail£º zhaofanqiu@gmail.com
** date£º 2016/07/08
** desc£º PRNNBase layer
*********************************************************************************/

#ifndef CAFFE_PRNN_BASE_LAYER_HPP_
#define CAFFE_PRNN_BASE_LAYER_HPP_

#include <string>
#include <utility>
#include <vector>

#include "boost/scoped_ptr.hpp"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/slice_layer.hpp"
#include "caffe/layers/concat_layer.hpp"

#include "caffe/layers/reshape_layer.hpp"
#include "caffe/layers/inner_product_layer.hpp"

namespace caffe {
	/**
	* @brief Implementation of PLSTM, used to do prediction
	*/
	template <typename Dtype>
	class PRNNBaseLayer : public Layer<Dtype> {
	public:
		explicit PRNNBaseLayer(const LayerParameter& param)
			: Layer<Dtype>(param) {}

		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		/*
		 * only h0_
		 */
		virtual inline int ExactNumBottomBlobs() const { return 1; }
		virtual inline int ExactNumTopBlobs() const { return 1; }

	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom){
			/// only used to do prediction, so backward is not needed
			NOT_IMPLEMENTED;
		}

		virtual void RecurrentForward(const int t) = 0;
		virtual void ShareWeight() = 0;
		
		int hidden_dim_;
		/// single sequence length
		int L_;

		// number of sequences
		int T_;

		int output_dim_;

		int bias_term_;

		// slice_h_0_ layer
		shared_ptr<SliceLayer<Dtype> > slice_h0_;
		vector<shared_ptr<Blob<Dtype> > > H0_;

		// concat_y_ layer
		shared_ptr<ConcatLayer<Dtype> > concat_y_;

		// innerproduct layer to produce output
		shared_ptr<InnerProductLayer<Dtype> > ip_h_;
		vector<shared_ptr<Blob<Dtype> > > Y_;

		// hidden states
		vector<shared_ptr<Blob<Dtype> > > H_;

		// TODO: this start blob_ can not initialized with specific settings
		shared_ptr<Blob<Dtype> > start_blob_;
	};
}  // namespace caffe

#endif  // CAFFE_RNN_BASE_LAYER_HPP_
