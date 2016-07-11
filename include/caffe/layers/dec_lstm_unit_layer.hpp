/********************************************************************************
** Copyright(c) 2016 USTC All Rights Reserved.
** auth£º Xu Shen
** mail£º shenxuustc@gmail.com
** date£º 2016/06/22
** desc£º DLSTMUnit layer
*********************************************************************************/
#ifndef CAFFE_DEC_LSTM_UNIT_LAYER_
#define CAFFE_DEC_LSTM_UNIT_LAYER_

#include <string>
#include <utility>
#include <vector>

#include "boost/scoped_ptr.hpp"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/layer.hpp"
#include "caffe/layers/drnn_base_layer.hpp"

namespace caffe{
	/**
	 * @brief A helper for DLSTMLayer: computes a single timestep of the 
	 * non-linearity of the DLSTM, produce the updated cell and hidden states.
	 * because c_0 and h_0 are passed from encoding LSTM, so we don't need to 
	 * reset c_0 and h_0 here
	 **/
	template <typename Dtype>
	class DLSTMUnitLayer : public Layer<Dtype>{
	public:
		explicit DLSTMUnitLayer<Dtype>(const LayerParameter& param)
			: Layer<Dtype>(param){}

		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual inline const char* type() const { return "DLSTMUnit"; }

		//c_{t-1}(D), X_{t-1}(4D)
		virtual inline int ExactNumBottomBlobs() const { return 2; }
		virtual inline int ExactNumTopBlobs() const { return 2; }

	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
		virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

		/// @brief The hidden and output dimension.
		int hidden_dim_;
		Blob<Dtype>	X_acts_;
	};
}

#endif