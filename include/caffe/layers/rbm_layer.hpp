#ifndef CAFFE_RBM_LAYER_HPP_
#define CAFFE_RBM_LAYER_HPP_

#include "boost/scoped_ptr.hpp"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/filler.hpp"
#include "caffe/layers/inner_product_layer.hpp"
#include "caffe/layers/sampling_layer.hpp"
#include "caffe/layers/sigmoid_layer.hpp"

namespace caffe{
	template <typename Dtype>
	class RBMLayer :public Layer<Dtype>{
	public:
		explicit RBMLayer(const LayerParameter& param)
			: Layer<Dtype>(param){}
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual inline const char* type() const { return "RBM"; }
		virtual inline int ExactNumBottomBlobs() const { return 1; }
		virtual inline int MinNumTopBlobs() const { return 1; }
		virtual inline int MaxNumTopBlobs() const { return 2; }

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

		virtual void ShareWeight(){
			ip_forward_layer_->blobs()[0]->ShareData(*(this->blobs_[0]));
			ip_back_layer_->blobs()[0]->ShareData(*(this->blobs_[0]));
			if (bias_term_){
				ip_forward_layer_->blobs()[1]->ShareData(*(this->blobs_[1]));
				ip_back_layer_->blobs()[1]->ShareData(*(this->blobs_[2]));
			}
		}

		void Gibbs_vhvh();

		//iteration times in contrasitive divergence
		int num_iter_;

	public:
		//visible variables
		shared_ptr<Blob<Dtype> > pos_v_;
		shared_ptr<Blob<Dtype> > neg_v_;

		//hidden variables
		shared_ptr<Blob<Dtype> > pos_h_;
		shared_ptr<Blob<Dtype> > neg_h_;

		//sampling result of corresponding variables
		shared_ptr<Blob<Dtype> > h_state_;
		shared_ptr<Blob<Dtype> > v_state_;

		// layer for linear computation
		/// vis to hidden
		shared_ptr<InnerProductLayer<Dtype> > ip_forward_layer_;
		/// hidden to vis
		shared_ptr<InnerProductLayer<Dtype> > ip_back_layer_;

		/* Layer for activation
		 * TODO: support for more activation types, e.g., ReLU, Tanh
		 */
		shared_ptr<SigmoidLayer<Dtype> > act_layer_;

		// layer for sampling
		shared_ptr<SamplingLayer<Dtype> > sample_layer_;

		//weight diff buffer
		Blob<Dtype>* weight_diff_buf_;
		int M_;
		int N_;
		int K_;
		bool bias_term_;
		// learn by unsupervised loss (CD)
		bool learn_by_cd_;
		// learn by supervised loss (top error)
		bool learn_by_top_;
		Blob<Dtype>* bias_multiplier_;
	};
} // namespace caffe
#endif