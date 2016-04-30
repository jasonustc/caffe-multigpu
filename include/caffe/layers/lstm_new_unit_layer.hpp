#ifndef CAFFE_LSTM_NEW_UNIT_LAYER_HPP_
#define CAFFE_LSTM_NEW_UNIT_LAYER_HPP_

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe{
	/**
	 * @brief A new helper for LSTMNewLayer: computes a single timestep of the 
	 *        non-linearity of the LSTM, producing the updated fast and slow memory
	 *        and hidden states.
	 **/
	template <typename Dtype>
	class LSTMNewUnitLayer : public Layer<Dtype>{
	public:
		explicit LSTMNewUnitLayer(const LayerParameter& param)
			: Layer<Dtype>(param){}
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual inline const char* type() const { return "LSTMNewUnit"; }

		// c_fast, c_slow, h, x
		virtual inline int ExactNumBottomBlobs() const { return 4; }
		virtual inline int ExactNumTopBlobs() const { return 3; }

		virtual inline bool AllowForceBackward(const int bottom_index) const{
			// Can't propagate to sequence continuation indicators.
			return bottoM_index != 2;
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

		// @brief The hidden and output dimension.
		int hidden_dim_;
		Blob<Dtype> X_acts_;
	};
}

#endif // CAFFE_LSTM_NEW_UNIT_LAYER_HPP_