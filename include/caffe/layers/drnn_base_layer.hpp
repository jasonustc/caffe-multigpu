/********************************************************************************
** Copyright(c) 2016 USTC All Rights Reserved.
** auth£º Xu Shen
** mail£º shenxuustc@gmail.com
** date£º 2016/06/22
** desc£º DRNNBase layer
*********************************************************************************/

#ifndef CAFFE_DRNN_BASE_LAYER_HPP_
#define CAFFE_DRNN_BASE_LAYER_HPP_

#include <string>
#include <utility>
#include <vector>

#include "caffe/util/db.hpp"
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

namespace caffe{
	/*
	 * for implementation of DLSTM
	 * input:
	 *      c_init_, h_init_, cont_, X_ (if conditional)
	 * here cont_ is used to infer the decoding sequence length, this
	 * is needed to deal with variour sequence length in X_
	 */
	template <typename Dtype>
	class DRNNBaseLayer : public Layer<Dtype>{
	public:
		explicit DRNNBaseLayer(const LayerParameter& param)
			: Layer<Dtype>(param){}

		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual inline int MinBottomBlobs() const { return 3; }
		virtual inline int MaxBottomBlobs() const { return 4; }
		virtual inline int ExactNumTopBlobs() const { return 1; }

		virtual inline bool AllowForceBackward(const int bottom_index) const{
			// Can't propagate to sequence continuation indicators.
			return bottom_index != 2;
		}

	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down,
			const vector<Blob<Dtype>*>& bottom);

		virtual inline void InferSeqLengths(Blob<Dtype>* cont);

		virtual void RecurrentForward(const int t) = 0;
		virtual void RecurrentBackward(const int t) = 0;
		virtual void ReorderDecodeInput(const vector<Blob<Dtype>*>& bottom);
		virtual void ReorderDecodeOutpout(const Blob<Dtype>* cont);
		virtual void ShareWeight() = 0;
		//NOTE: maybe not needed
		virtual int GetHiddenDim() = 0;

		int hidden_dim_;
		//number of sequences
		int num_seq_;
		int T_;
		bool reverse_;
		bool conditional_;

		// to infer length of seqences from cont_
		// NOTE: may not need
		vector<int> seq_lens_;

		// slice_h_ layer
		shared_ptr<SliceLayer<Dtype> > slice_h_;
		vector<shared_ptr<Blob<Dtype> > > H0_;

		// slice_c_ layer
		shared_ptr<SliceLayer<Dtype> > slice_c_;
		vector<shared_ptr<Blob<Dtype> > > C0_;

		// slice_x_ layer
		shared_ptr<SliceLayer<Dtype> > slice_x_;
		vector<shared_ptr<Blob<Dtype> > > X_;

		// concat_h_ layer
		shared_ptr<ConcatLayer<Dtype> > concat_h_dec_;
		vector<shared_ptr<Blob<Dtype> > > H_DEC_;
		
		// internal split layers
		vector<shared_ptr<SplitLayer<Dtype> > > split_layers_;

		// To store the buffer of the output
		vector<shared_ptr<Blob<Dtype> > > decode_output_;
		vector<shared_ptr<Blob<Dtype> > > decode_input_;
		shared_ptr<Blob<Dtype> > zero_blob_;
	};
}

#endif // CAFFE_DRNN_BASE_LAYER_HPP_
