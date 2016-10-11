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
#include "caffe/layers/scale_layer.hpp"
#include "caffe/util/rng.hpp"

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
		// used to block partial of the data in visible input
		Blob<Dtype>* v_mask_;
		// used to multiply v by v_mask_
		shared_ptr<ScaleLayer<Dtype> > scale_layer_;
		// the start point of blocked feat
		int block_start_;
		// the end point of blocked feat
		int block_end_;
		// indicator
		bool block_feat_;
		/**
		 * @brief Generates a random integer from {0, 1, ..., n - 1}.
		 * @param n the upper bound (exclusive) value of the value number.
		 * @return int
		 **/
		virtual int Rand(int n){
			CHECK(rng_);
			CHECK_GT(n, 0);
			caffe::rng_t* rng =
				static_cast<caffe::rng_t*>(rng_->generator());
			return ((*rng)() % n);
		}

		shared_ptr<Caffe::RNG> rng_;

		// if we need to randomly block features
		bool random_block_;
	};
} // namespace caffe
#endif