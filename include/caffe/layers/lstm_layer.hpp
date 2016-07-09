#ifndef CAFFE_LSTM_LAYER_HPP_
#define CAFFE_LSTM_LAYER_HPP_

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

#include "caffe/layers/rnn_base_layer.hpp"
#include "caffe/layers/lstm_unit_layer.hpp"

#include "caffe/layers/slice_layer.hpp"
#include "caffe/layers/split_layer.hpp"
#include "caffe/layers/concat_layer.hpp"
#include "caffe/layers/scale_layer.hpp"
#include "caffe/layers/inner_product_layer.hpp"

namespace caffe {
	/**
	* @brief Implementation of LSTM
	*/
	template <typename Dtype>
	class LSTMLayer : public RNNBaseLayer<Dtype> {
	public:
		explicit LSTMLayer(const LayerParameter& param)
			: RNNBaseLayer<Dtype>(param) {}

		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual inline const char* type() const { return "LSTM"; }

		virtual vector<Blob<Dtype>*> RecurrentOutput()
		{
			vector<Blob<Dtype>*> output{
				H0_.get(),
				C0_.get()
			};
			return output;
		}


	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
		virtual void RecurrentForward(const int t);
		virtual void RecurrentBackward(const int t);
		virtual void CopyRecurrentOutput()
		{
			if (Caffe::mode() == Caffe::GPU) {
				caffe_copy(H0_->count(), H_[T_ - 1]->mutable_gpu_data(),
					H0_->mutable_gpu_data());
				caffe_copy(H0_->count(), C_[T_ - 1]->mutable_gpu_data(),
					C0_->mutable_gpu_data());
			}
			else
			{
				caffe_copy(H0_->count(), H_[T_ - 1]->mutable_cpu_data(),
					H0_->mutable_cpu_data());
				caffe_copy(H0_->count(), C_[T_ - 1]->mutable_cpu_data(),
					C0_->mutable_cpu_data());
			}
		}
		virtual void ShareWeight()
		{
			ip_g_->blobs()[0]->ShareData(*(blobs_[0]));
			ip_g_->blobs()[0]->ShareDiff(*(blobs_[0]));
			if (bias_term_)
			{
				ip_g_->blobs()[1]->ShareData(*(blobs_[1]));
				ip_g_->blobs()[1]->ShareDiff(*(blobs_[1]));
			}
		}
		virtual int GetHiddenDim()
		{
			return this->layer_param_.inner_product_param().num_output();
		}

		int bias_term_;
		bool out_ct_;

		//Data blobs
		shared_ptr<Blob<Dtype> > C0_;
		shared_ptr<Blob<Dtype> > H0_;
		
		//Layers
		// split_h_ layer
		// split LSTMUnit output (h_1,h_2,..., h_T)
		shared_ptr<SplitLayer<Dtype> > split_h_;
		vector<shared_ptr<Blob<Dtype> > > H_1_;
		vector<shared_ptr<Blob<Dtype> > > H_2_;

		// scale_h_ layer
		shared_ptr<ScaleLayer<Dtype> > scale_h_;
		vector<shared_ptr<Blob<Dtype> > > SH_;

		// concat_h_ layer
		shared_ptr<ConcatLayer<Dtype> > concat_;
		vector<shared_ptr<Blob<Dtype> > > XH_;

		// ip_g_ layer
		shared_ptr<InnerProductLayer<Dtype> > ip_g_;
		vector<shared_ptr<Blob<Dtype> > > G_;

		// lstm_unit_h_ layer
		shared_ptr<LSTMUnitLayer<Dtype> > lstm_unit_;
		vector<shared_ptr<Blob<Dtype> > > C_;

		// concat_ct_ layer, used for decoding LSTM
		shared_ptr<ConcatLayer<Dtype> > concat_ct_;
		shared_ptr<SplitLayer<Dtype> > split_c_;
		vector<shared_ptr<Blob<Dtype> > > C_1_;
		vector<shared_ptr<Blob<Dtype> > > C_2_;
	};
}  // namespace caffe

#endif  // CAFFE_LSTM_LAYER_HPP_
