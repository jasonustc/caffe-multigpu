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

#include "caffe/layers/lstm_layer.hpp"
#include "caffe/layers/lstm_unit_layer.hpp"

#include "caffe/layers/slice_layer.hpp"
#include "caffe/layers/split_layer.hpp"
#include "caffe/layers/concat_layer.hpp"
#include "caffe/layers/scale_layer.hpp"
#include "caffe/layers/inner_product_layer.hpp"
#include "caffe/layers/relu_layer.hpp"
#include "caffe/layers/sigmoid_layer.hpp"
#include "caffe/layers/euclidean_loss_layer.hpp"

namespace caffe{
	template <typename Dtype>
	class LocalLSTMLayer : public LSTMLayer<Dtype>{
	public: 
		explicit LocalLSTMLayer(const LayerParameter& param)
			: LSTMLayer<Dtype>(param){}
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual inline const char* type() const { return "LocalLSTM"; }

	protected:
		virtual void RecurrentForward(const int t);
		virtual void RecurrentBackward(const int t);
		virtual void LocalUpdateRecurrent(const int t);
		void Regularize(const Dtype local_decay, const int id);
		void ClipGradients();
		void ComputeUpdateValue(const Dtype lr, const Dtype mom, const int id);
		void ClearLocalParamDiffs();

		// ip_hp_ layer
		/// innerproduct layer to predict the input
		shared_ptr<InnerProductLayer<Dtype> > ip_xp_;
		/// activation layer
		shared_ptr<Layer<Dtype> > act_layer_;
		/// prediction of next input 
		shared_ptr<Blob<Dtype> > px_;

		// local loss layer
		shared_ptr<Layer<Dtype> > loss_layer_;
		shared_ptr<Blob<Dtype> > local_loss_;

		Dtype local_lr_;
		// how much does the local lr decay through time step?
		Dtype local_lr_decay_;
		Dtype local_decay_;
		Dtype local_gradient_clip_;
		bool local_bias_term_;
		Dtype local_momentum_;
		string regularize_type_;
		int back_steps_;


		// temp_ for L1 decay and history
		vector<shared_ptr<Blob<Dtype> > > temp_;

		// params need to be updated in local learning
		vector<shared_ptr<Blob<Dtype> > > local_learn_params_;
	};

} // namespace caffe
#endif
