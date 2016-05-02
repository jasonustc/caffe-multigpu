/********************************************************************************
** Copyright(c) 2015 USTC Reserved.
** auth: Xu Shen
** mail: shenxu@mail.ustc.edu.cn
** date: 2015/11/4
** desc: RandomTransformLayer(GPU)
*********************************************************************************/
#ifndef CAFFE_RANDOM_TRANSFORM_LAYER_HPP_
#define CAFFE_RANDOM_TRANSFORM_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/transform.hpp"
#include "caffe/util/rng.hpp"

namespace caffe{
	/**
	 * @brief randomly transform input by 2-D transformations
	 *        and generate output with the same size by cropping or padding
	 **/

	template <typename Dtype>
	class RandomTransformLayer : public Layer<Dtype>{
	public:
		explicit RandomTransformLayer(const LayerParameter& param) : Layer<Dtype>(param){}
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);

		virtual inline const char* type() const { return "random_transform"; }

		/**
		 * @brief Initialize the Random number generations if needed
		 **/
		void InitRand(){
			needs_rand_ = this->layer_param_.rand_trans_param().alternate() && 
				(this->phase_ == TRAIN);
			if (needs_rand_){
				const unsigned int rng_seed = caffe_rng_rand();
				rng_.reset(new Caffe::RNG(rng_seed));
			}
			else{
				rng_.reset();
			}
		}

	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<bool>& propagate_down,
			const vector<Blob<Dtype>*>& top);
		virtual void Backward_gpu(const vector<Blob<Dtype>*>& bottom,
			const vector<bool>& propagate_down,
			const vector<Blob<Dtype>*>& top);


		virtual inline bool EqualNumBottomTopBlobs() const { return true; }
		virtual inline int ExtactNumBottomBlobs() const { return 1; }
		virtual inline int ExtactNumTopBlobs() const { return 1; }

		//added by xu shen here
		//get the coordination matrix after transformation
		void GetTransCoord_cpu();
		void GetTransCoord_gpu();

		bool rotation_;
		bool scale_;
		bool shift_;

		int Height_;
		int Width_;

		Dtype start_angle_;
		Dtype end_angle_;
		Dtype start_scale_;
		Dtype end_scale_;
		Dtype dx_prop_;
		Dtype dy_prop_;

		Dtype curr_scale_;
		Dtype curr_angle_;
		//shift by #pixels
		Dtype curr_shift_x_;
		Dtype curr_shift_y_;

		//std of scale, angle, shift_x, shift_y sampling
		Dtype std_scale_;
		Dtype std_angle_;
		Dtype std_dx_prop_;
		Dtype std_dy_prop_;
		//clip of scale and shift in the transform
		Dtype min_scale_;
		Dtype max_scale_;
		Dtype max_shift_prop_;

		Border BORDER_; // border type
		Interp INTERP_; //interpolation type
		//sampling type for transform parameters
		RandTransformParameter_SampleType sample_type_;

		//3x3 transform matrix buffer, row order
		Blob<Dtype> tmat_;

		//Indices for image transformation
		//We use blob's data to be fwd and diff to be backward indices
		Blob<Dtype> coord_idx_;
		//here to store the original coord_
		Blob<Dtype> original_coord_;

	protected:
		/**
		 * @brief Generates a random integer from Uniform({0, 1, ..., n-1}). 
		 *
		 * @param n
		 * The upper bound (exclusive) value of the random number.
		 * @return 
		 * A uniformly random integer value from ({0, 1, ..., n-1}).
		 **/
		virtual int Rand(int n){
			CHECK(rng_);
			CHECK_GT(n, 0);
			caffe::rng_t* rng =
				static_cast<caffe::rng_t*>(rng_->generator());
			return ((*rng)() % n);
		}

		//initialize the transform matrix to identity
		void InitTransform(){
			Dtype* tmat_data = tmat_.mutable_cpu_data();
			std::fill(tmat_data, tmat_data + 9, 0);
			tmat_data[0] = tmat_data[4] = tmat_data[8] = 1;
		}

		shared_ptr<Caffe::RNG> rng_;
		bool needs_rand_;
		int rand_;

		//indicators in a single forward pass for random transformations
		bool need_scale_;
		bool need_rotation_;
		bool need_shift_;
	};
}

#endif
