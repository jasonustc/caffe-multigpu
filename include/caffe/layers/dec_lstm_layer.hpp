/********************************************************************************
** Copyright(c) 2016 USTC All Rights Reserved.
** auth£º Xu Shen
** mail£º shenxuustc@gmail.com
** date£º 2016/06/22
** desc£º DLSTM layer
*********************************************************************************/
#ifndef CAFFE_DEC_LSTM_LAYER_HPP_
#define CAFFE_DEC_LSTM_LAYER_HPP_

#include "caffe/util/db.hpp"
#include "boost/scoped_ptr.hpp"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/net.hpp"

#include "caffe/layers/drnn_base_layer.hpp"
#include "caffe/layers/dec_lstm_unit_layer.hpp"

#include "caffe/layers/slice_layer.hpp"
#include "caffe/layers/split_layer.hpp"
#include "caffe/layers/concat_layer.hpp"
#include "caffe/layers/scale_layer.hpp"
#include "caffe/layers/inner_product_layer.hpp"

namespace caffe{
	/*
	 * @brief Implementation of Decoding LSTM
	 * TODO: deal with param_propagate_down_ and propagate_down
	 * if decoding a sentence, the input is X_[0], X_[1], ...
	 * if decoding a video, the input is \hat{0}, X_[0], X_[1], ...
	 * if c0_id is specified, we use external c0s(bottom[c0_id]) as initial memory cell
	 * otherwise, we use zero c0s
	 */
	template <typename Dtype>
	class DLSTMLayer : public DRNNBaseLayer<Dtype>{
	public:
		explicit DLSTMLayer(const LayerParameter& param)
			: DRNNBaseLayer<Dtype>(param){}
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual inline const char* type() const { return "DLSTM"; }


	protected:

		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down,
			const vector<Blob<Dtype>*>& bottom);

		virtual void RecurrentForward(const int t, const int cont_t, const int seq_id);
		virtual void RecurrentBackward(const int t, const int cont_t, const int seq_id);

		virtual void ShareWeight(){
			this->ip_h_->blobs()[0]->ShareData(*(this->blobs_[0]));
			this->ip_h_->blobs()[0]->ShareDiff(*(this->blobs_[0]));
			ip_g_->blobs()[0]->ShareData(*(this->blobs_[1]));
			ip_g_->blobs()[0]->ShareDiff(*(this->blobs_[1]));
			if (bias_term_){
				this->ip_h_->blobs()[1]->ShareData(*(this->blobs_[2]));
				this->ip_h_->blobs()[1]->ShareDiff(*(this->blobs_[2]));
				ip_g_->blobs()[1]->ShareData(*(this->blobs_[3]));
				ip_g_->blobs()[1]->ShareDiff(*(this->blobs_[3]));
			}
		}

		// just copy data, not diff
		virtual void CopyRecurrentOutputAndInput(){
			if (Caffe::mode() == Caffe::GPU) {
				caffe_copy(this->start_H_->count(), this->H_[this->T_ - 1]->mutable_gpu_data(),
					this->start_H_->mutable_gpu_data());
				caffe_copy(this->start_H_->count(), C_[this->T_ - 1]->mutable_gpu_data(),
					start_C_->mutable_gpu_data());
				if (this->delay_){
					Blob<Dtype>* input_blob = this->conditional_ ? this->X_[this->T_ - 1].get()
						: this->Y_2_[this->T_ - 1].get();
					caffe_copy(input_blob->count(), input_blob->gpu_data(), 
						this->start_blob_->mutable_gpu_data());
				}
			}
			else
			{
				caffe_copy(this->start_H_->count(), this->H_[this->T_ - 1]->mutable_cpu_data(),
					this->start_H_->mutable_cpu_data());
				caffe_copy(this->start_H_->count(), C_[this->T_ - 1]->mutable_cpu_data(),
					start_C_->mutable_cpu_data());
				if (this->delay_){
					Blob<Dtype>* input_blob = this->conditional_ ? this->X_[this->T_ - 1].get()
						: this->Y_2_[this->T_ - 1].get();
					caffe_copy(input_blob->count(), input_blob->cpu_data(), 
						this->start_blob_->mutable_cpu_data());
				}
			}
		}

		int bias_term_;
		bool has_c0_;
		// external c0 id
		int c0_id_;

		//Layers
		// split_h_ layer
		// split LSTMUnit output (h_1, h_2, ..., h_T)
		// one for output and one for input of next cell
		shared_ptr<SplitLayer<Dtype> > split_h_;
		vector<shared_ptr<Blob<Dtype> > > H_1_;
		vector<shared_ptr<Blob<Dtype> > > H_2_;

		// slice_c_ layer
		shared_ptr<SliceLayer<Dtype> > slice_c_;
		vector<shared_ptr<Blob<Dtype> > > C0_;

		// if not conditional_, we need to feed h into next cell as input

		// concat_h_ layer
		// concat i_x and i_h
		shared_ptr<ConcatLayer<Dtype> > concat_;
		vector<shared_ptr<Blob<Dtype> > > XH_;

		// ip_g_ layer
		shared_ptr<InnerProductLayer<Dtype> > ip_g_;
		vector<shared_ptr<Blob<Dtype> > > G_;

		// dlstm_unit_h_ layer
		shared_ptr<DLSTMUnitLayer<Dtype> > dlstm_unit_;
		vector<shared_ptr<Blob<Dtype> > > C_;

		// c for t == 0 if needed
		shared_ptr<Blob<Dtype> > start_C_;
	};
}

#endif
