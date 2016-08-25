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

		void Gibbs_vhvh_cpu();
		void Gibbs_vhvh_gpu();

		//iteration times in contrasitive divergence
		int num_iteration_;

	private:
		//visible variables
		shared_ptr<Blob<Dtype> > pos_v_;
		shared_ptr<Blob<Dtype> > neg_v_;

		//hidden variables
		shared_ptr<Blob<Dtype> > pos_h_;
		shared_ptr<Blob<Dtype> > neg_h_;

		//sampling result of positive hidden states
		shared_ptr<Blob<Dtype> > pos_state_h_;
		shared_ptr<Blob<Dtype> > neg_state_v_;

		bool bias_term_;
		Blob<Dtype>* bias_multiplier_;

		RBMParameter_SampleType sample_type_;

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
		// what is this for?
		Blob<Dtype>* weight_diff_buf_;
	};
} // namespace caffe
#endif