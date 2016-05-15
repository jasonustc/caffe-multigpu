#ifndef CAFFE_SI_LAYERS_HPP_
#define CAFFE_SI_LAYERS_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/transformation.hpp"

namespace caffe{
	// AJ: Upsampling layer, that takes in one bottom and returns #
	// transformations+1 many tops
	template <typename Dtype> class UpsamplingLayer : public Layer<Dtype> {
	public:
		explicit UpsamplingLayer(const LayerParameter &param) : Layer<Dtype>(param) {}
		// Not adhereing with the LayerSetup/Reshape setup for this layer because
		// Reshape will recompute the transformation indices and it's waste of
		// computation. Keep reshape empty & do everything in LayerSetUp.
		virtual void LayerSetUp(const vector<Blob<Dtype> *> &bottom,
			const vector<Blob<Dtype> *>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top) {};

		virtual inline const char* type() const {
			return "Up";
		}
		virtual inline int MinBottomBlobs() const { return 1; }
		virtual inline int MinTopBlobs() const { return 1; }

	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype> *> &bottom,
			const vector<Blob<Dtype> *> &top);
		virtual void Forward_gpu(const vector<Blob<Dtype> *> &bottom,
			const vector<Blob<Dtype> *>& top);
		virtual void Backward_cpu(const vector<Blob<Dtype> *> &top,
			const vector<bool>& propagate_down,
			const vector<Blob<Dtype>*>& bottom);
		virtual void Backward_gpu(const vector<Blob<Dtype> *> &top,
			const vector<bool>& propagate_down,
			const vector<Blob<Dtype> *>& bottom);

		int NUM_T_;
		int HEIGHT_;    // of input
		int WIDTH_;     // of input
		Border BORDER_; // of transformation
		Interp INTERP_; // of transformation
		// 3x3 rotation matrix buffer, row ordater:
		float tmat_[9];
		// Indices for image transformation
		// We use blob's data to be fwd and diff to be backward indices
		vector<shared_ptr<Blob<float> > > coord_idx_;

		// bc TIConvolutionLayer has to call F/Bward_g/cpu functions
		template <typename D> friend class TIConvolutionLayer;
	};


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

	// AJ: DownPooling layer, that takes in multiple bottoms and returns 1 top
	// reinterpolate all the bottom into a canonical shape (shape of bottom[0])
	// then computes max/ave pooling across all bottoms (transformations)
	template <typename Dtype> class DownPoolingLayer : public Layer<Dtype> {
	public:
		explicit DownPoolingLayer(const LayerParameter &param)
			: Layer<Dtype>(param), switch_idx_(0, 0, 0, false) {}
		// Same as Upsample, Reshape does nothing, everythin is done in LayerSetUp.
		virtual void LayerSetUp(const vector<Blob<Dtype> *> &bottom,
			const vector<Blob<Dtype> *> &top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*> &top) {};

		virtual inline const char* type() const {
			return "Downpooling";
		}
		virtual inline int MinBottomBlobs() const { return 1; }
		virtual inline int ExactNumTopBlobs() const { return 1; }

		virtual void Report(const std::string &name);

		const Blob<float> &max_switch() { return switch_idx_; }
		const vector<shared_ptr<Blob<float> > > &coord_idx() { return coord_idx_; }

		inline const vector<float> &trans_counter() { return trans_counter_; }

	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype> *> &bottom,
			const vector<Blob<Dtype> *> &top);
		virtual void Forward_gpu(const vector<Blob<Dtype> *> &bottom,
			const vector<Blob<Dtype> *> &top);
		virtual void Backward_cpu(const vector<Blob<Dtype> *> &top,
			const vector<bool>& propagate_down,
			const vector<Blob<Dtype> *>& bottom);
		virtual void Backward_gpu(const vector<Blob<Dtype> *> &top,
			const vector<bool>& propagate_down,
			const vector<Blob<Dtype> *> &bottom);
		void UpdateCounter(const int *curr_counter, const int &top_count);

		int NUM_T_;
		int HEIGHT_; // of output
		int WIDTH_;  // of output
		int NUM_OUTPUT_;
		int CHANNEL_OUTPUT_;
		Border BORDER_; // of transformation
		Interp INTERP_; // of transformation

		// 3x3 rotation matrix buffer:
		float tmat_[9];
		// Indices for image transformation
		// We use blob's data to be fwd and diff to be backward indices
		vector<shared_ptr<Blob<float> > > coord_idx_;
		// Switches for max-pooling over transformations
		Blob<float> switch_idx_;
		// TODO(AJ): Since top sizes are all going to be the same we don't need to
		// have this top_buffer_ as a vector
		vector<shared_ptr<Blob<Dtype> > > top_buffer_;

		// For debugging/printing how often each transformation was used
		vector<float> trans_counter_;

		// bc TIConvolutionLayer has to call F/Bward_g/cpu functions
		template <typename D> friend class TIConvolutionLayer;
	};


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
} // namespace caffe
#endif
