#ifndef CAFFE_LSTM_NEW_LAYER_HPP_
#define CAFFE_LSTM_NEW_LAYER_HPP_

#include <string>
#include <utility>
#include <vector>

#include "boost/shared_ptr.hpp"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/rnn_base_layer.hpp"
#include "caffe/layers/lstm_new_unit_layer.hpp"

#include "caffe/layers/slice_layer.hpp"
#include "caffe/layers/split_layer.hpp"
#include "caffe/layers/concat_layer.hpp"
#include "caffe/layers/scale_layer.hpp"
#include "caffe/layers/inner_product_layer.hpp"

namespace caffe{
	/**
	 * @brief Implementation of LSTM
	 */
	template <typename Dtype>
	class LSTMNewLayer : public RNNBaseLayer<Dtype>{
	public:
		explicit LSTMNewLayer(const LayerParameter& param)
			: RNNBaseLayer<Dtype>(param){}

		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual inline const char* type() const { return "LSTMNew"; }

		virtual vector<Blob<Dtype>*> RecurrentOutput(){
			vector<Blob<Dtype>*> output(
				H0_.get();
				C_fast0_.get();
				C_slow0_.get();
			);
			return output;
		}
	};
}

#endif // CAFFE_LSTM_NEW_LAYER_HPP_