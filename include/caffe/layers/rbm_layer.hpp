#ifndef CAFFE_RBM_LAYER_HPP_
#define CAFFE_RBM_LAYER_HPP_

#include "boost/scoped_ptr.hpp"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/filler.hpp"

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

		//the inner product parameters
		int M_;
		int K_;
		int N_;
		//iteration times in contrasitive divergence
		int num_iteration_;

	public:
		//visible variables
		Blob<Dtype> pos_v_;
		Blob<Dtype> neg_v_;

		//hidden variables
		Blob<Dtype> pos_h_;
		Blob<Dtype> neg_h_;

		//sampling result of positive hidden states
		Blob<Dtype> positive_state_h_;
		Blob<Dtype> negative_state_v_;

		bool bias_term_;
		Blob<Dtype> bias_multiplier_;

		RBMParameter_SampleType sample_type_;

		//weight diff buffer
		Blob<Dtype> weight_diff_buf_;
	};
} // namespace caffe
#endif