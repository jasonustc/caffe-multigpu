#ifndef CAFFE_TICONV_LAYER_HPP_
#define CAFFE_TICONV_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/layers/up_layer.hpp"
#include "caffe/layers/tiedconv_layer.hpp"
#include "caffe/layers/downpool_layer.hpp"

namespace caffe{
	// AJ: Transformation invariant convolution layer, wrapper around UP, TiedConv,
	// and Down
	template <typename Dtype> 
	class TIConvolutionLayer : public Layer<Dtype> {
	public:
		explicit TIConvolutionLayer(const LayerParameter &param)
			: Layer<Dtype>(param) {
				up_layer_ = NULL;
				tiedconv_layer_ = NULL;
				downpool_layer_ = NULL;
			}
		virtual ~TIConvolutionLayer() {
			delete up_layer_;
			delete tiedconv_layer_;
			delete downpool_layer_;
		}

		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top) {};

		virtual inline const char* type() const {
			return "TiConv";
		}
		virtual inline int ExactBottomBlobs() const { return 1; }
		virtual inline int ExactNumTopBlobs() const { return 1; }
		virtual inline bool EqualNumBottomTopBlobs() const { return true; }

		virtual void Report(const std::string &name);
		vector<shared_ptr<Blob<Dtype> > > &activations() { return activations_; }
		Blob<float> &max_switch() { return downpool_layer_->switch_idx_; }

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
			const vector<Blob<Dtype> *>& bottom);

		// same as the blobs_ in net.hpp, contains the output of the uplayer
		// and the output of the tiedconv layer. i.e. size = 2*NumT
		// this is NOT the blob_
		vector<shared_ptr<Blob<Dtype> > > activations_;

		vector<Blob<Dtype> *> up_top_vec_;
		vector<Blob<Dtype> *> tiedconv_top_vec_;

		// the wrapped layers:
		UpsamplingLayer<Dtype> *up_layer_;
		TiedConvolutionLayer<Dtype> *tiedconv_layer_;
		DownPoolingLayer<Dtype> *downpool_layer_;

		int NUM_T_;
	};
}
#endif
