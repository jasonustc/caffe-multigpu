#ifndef CAFFE_SWITCH_LAYER_
#define CAFFE_SWITCH_LAYER_

#include <vector>
#include <utility>

#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/blob.hpp"
#include "caffe/util/rng.hpp"

namespace caffe{
	/*
	 * @brief iteratively or randomly output one of the inputs
	 */
	template <typename Dtype>
	class SwitchLayer : public Layer<Dtype>{
	public:
		explicit SwitchLayer(const LayerParameter& param)
		: Layer<Dtype>(param){}
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual inline int MinBottomBlobs() const { return 1; }
		virtual inline int MinTopBlobs() const { return 1; }
		virtual inline const char* type() const { return "Switch"; }
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
		int bottom_index_;
		bool has_bottom_index_;
	};

} // namespace caffe

#endif