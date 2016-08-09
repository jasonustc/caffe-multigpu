/********************************************************************************
** Copyright(c) 2016 USTC All Rights Reserved.
** auth£º Xu Shen
** mail£º shenxuustc@gmail.com
** date£º 2016/08/08
** desc£º LocalLSTM layer
*********************************************************************************/
#ifndef CAFFE_LOCAL_LSTM_LAYER_HPP_
#define CAFFE_LOCAL_LSTM_LAYER_HPP_

#include "caffe/util/db.hpp"
#include "boost/scoped_ptr.hpp"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/net.hpp"

#include "caffe/layers/rnn_base_layer.hpp"
#include "caffe/layers/lstm_unit_layer.hpp"

#include "caffe/layers/slice_layer.hpp"
#include "caffe/layers/split_layer.hpp"
#include "caffe/layers/concat_layer.hpp"
#include "caffe/layers/scale_layer.hpp"
#include "caffe/layers/inner_product_layer.hpp"

namespace caffe{
	template <typename Dtype>
	class LocalLSTMLayer : public RNNBaseLayer<Dtype>{
	public: 
		explicit LocalLSTMLayer(const LayerParameter& param)
			: RNNBaseLayer<Dtype>(param){}
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual inline const char* type() const { return "LocalLSTM"; }

	protected:
		virtual void RecurrentForward(const int t);

		virtual void ShareWeight(){
			ip_g_->blobs()[0]->ShareData(*(blobs_[0]));
			ip_g_->blobs()[0]->ShareDiff(*(blobs_[0]));
			if (bias_term_)
			{
				ip_g_->blobs()[1]->ShareData(*(blobs_[1]));
				ip_g_->blobs()[1]->ShareDiff(*(blobs_[1]));
			}
		}

		// concat_h_ layer
		// concat i_x and i_h
		shared_ptr<ConcatLayer<Dtype> > concat_;
		vector<shared_ptr<Blob<Dtype> > > XH_;

		// ip_g_ layer
		shared_ptr<InnerProductLayer<Dtype> > ip_g_;
		vector<shared_ptr<Blob<Dtype> > > G_;

		// LocalLSTM_unit_h_ layer
		shared_ptr<LSTMUnitLayer<Dtype> > lstm_unit_;
		vector<shared_ptr<Blob<Dtype> > > C_;

		shared_ptr<Blob<Dtype> > C0_;

	};

} // namespace caffe
#endif
