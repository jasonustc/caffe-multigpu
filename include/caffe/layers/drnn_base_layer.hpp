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

#include "boost/scoped_ptr.hpp"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/slice_layer.hpp"
#include "caffe/layers/concat_layer.hpp"
#include "caffe/layers/inner_product_layer.hpp"
#include "caffe/layers/split_layer.hpp"

namespace caffe{
	/*
	 * for implementation of DLSTM
	 * input:
	 *      c_init_, h_init_, cont_, X_ (if conditional)
	 * TODO: deal with changed continuation indicators accross different batches
	 */
	template <typename Dtype>
	class DRNNBaseLayer : public Layer<Dtype>{
	public:
		explicit DRNNBaseLayer(const LayerParameter& param)
			: Layer<Dtype>(param){}

		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
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

		// the child class should implement this
		virtual void RecurrentForward(const int t, const int cont_t, const int seq_id) = 0;
		virtual void RecurrentBackward(const int t, const int cont_t, const int seq_id) = 0;
		virtual void ShareWeight() = 0;

		int hidden_dim_;
		int T_;
		bool conditional_;
		int output_dim_;
		int X_dim_;
		int num_seq_;

		// slice_h_ layer
		shared_ptr<SliceLayer<Dtype> > slice_h_;
		vector<shared_ptr<Blob<Dtype> > > H0_;

		// slice_c_ layer
		shared_ptr<SliceLayer<Dtype> > slice_c_;
		vector<shared_ptr<Blob<Dtype> > > C0_;

		// slice_x_ layer
		shared_ptr<SliceLayer<Dtype> > slice_x_;
		vector<shared_ptr<Blob<Dtype> > > X_;

		// concat_y_ layer
		shared_ptr<ConcatLayer<Dtype> > concat_y_;

		// innerproduct layer to produce output
		shared_ptr<InnerProductLayer<Dtype> > ip_h_;
		vector<shared_ptr<Blob<Dtype> > > Y_;
		
		// output predictions
		// if not conditional, we need to feed prediction of last 
		// time as current input
		shared_ptr<SplitLayer<Dtype> >  split_y_;
		vector<shared_ptr<Blob<Dtype> > > Y_1_;
		vector<shared_ptr<Blob<Dtype> > > Y_2_;

		// zero blob for the input of the beginning
		shared_ptr<Blob<Dtype> > zero_blob_;

		// hidden states
		vector<shared_ptr<Blob<Dtype> > > H_;
	};
}

#endif // CAFFE_DRNN_BASE_LAYER_HPP_
