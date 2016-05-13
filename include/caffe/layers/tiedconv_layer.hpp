#ifndef CAFFE_TIEDCONV_LAYER_HPP_
#define CAFFE_TIEDCONV_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe{
	// Scale-invariant CNN layers
	// AJ: TiedConvolutionLayer, takes in multiple inputs and multiple outputs
	template <typename Dtype> class TiedConvolutionLayer : public Layer<Dtype> {
	public:
		explicit TiedConvolutionLayer(const LayerParameter &param)
			: Layer<Dtype>(param) {}
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		inline int num_in() const { return num_in_; }
		virtual void Report(const std::string &name);

	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype> *> &bottom,
			const vector<Blob<Dtype> *>& top);
		virtual void Forward_gpu(const vector<Blob<Dtype> *> &bottom,
			const vector<Blob<Dtype> *>& top);
		virtual void Backward_cpu(const vector<Blob<Dtype> *> &top,
			const vector<bool>& propagate_down,
			const vector<Blob<Dtype> *>& bottom);
		virtual void Backward_gpu(const vector<Blob<Dtype> *> &top,
			const vector<bool>& propagate_down,
			const vector<Blob<Dtype> *> &bottom);
		// These are for the convolution
		int kernel_h_, kernel_w_;
		int stride_h_, stride_w_;
		int num_;
		int channels_;
		int pad_h_, pad_w_;
		int group_;
		int num_output_;
		bool bias_term_;
		int M_; // feature maps, same for all transf
		int K_; // size of kernel (kernel_size^2*#channels_input)
		// AJ new stuff
		int num_in_; // total # of of bottom/top
		// col_buffer_ is a vector of size NUM_IN, needed for each input
		vector<shared_ptr<Blob<Dtype> > > col_buffers_;
		vector<shared_ptr<SyncedMemory> > bias_multipliers_;
		vector<int> N_; // each transformation has its own height_out*width_out
		vector<int> height_;
		vector<int> width_;

		// bc TIConvolutionLayer has to call F/Bward_g/cpu functions
		template <typename D> friend class TIConvolutionLayer;
	};
}
#endif
