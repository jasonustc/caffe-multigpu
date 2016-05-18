/********************************************************************************
** Copyright(c) 2015 USTC Reserved.
** auth: Xu Shen
** mail: shenxu@mail.ustc.edu.cn
** date: 2015/11/4
** desc: RandomTransformLayer(GPU)
*********************************************************************************/
#include <vector>
#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/filler.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/random_transform_layer.hpp"

namespace caffe{
	template <typename Dtype>
	void RandomTransformLayer<Dtype>::GetTransCoord_gpu(){
		//here we use cpu to compute tranform matrix
		InitTransform();
		float* tmat_cpu_data = tmat_.mutable_cpu_data();
		switch (sample_type_){
		case RandTransformParameter_SampleType_UNIFORM:
			if (need_rotation_){
				//randomly generate rotation angle
				caffe_rng_uniform(1, start_angle_, end_angle_, &curr_angle_);
				TMatFromParam(ROTATION, curr_angle_, curr_angle_, tmat_cpu_data);
			}
			if (need_scale_){
				caffe_rng_uniform(1, start_scale_, end_scale_, &curr_scale_);
				TMatFromParam(SCALE, curr_scale_, curr_scale_, tmat_cpu_data);
			}
			if (need_shift_){
				float shift_pixels_x = dx_prop_ * Width_;
				float shift_pixels_y = dy_prop_ * Height_;
				caffe_rng_uniform(1, -shift_pixels_x, shift_pixels_x, &curr_shift_x_);
				caffe_rng_uniform(1, -shift_pixels_y, shift_pixels_y, &curr_shift_y_);
				TMatFromParam(SHIFT, curr_shift_x_, curr_shift_y_, tmat_cpu_data);
			}
			break;
		//TODO: check if the threshold of the parameters are reasonable
		case RandTransformParameter_SampleType_GAUSSIAN:
			if (need_rotation_){
				//clip to in [-180, 180]
				caffe_rng_gaussian(1, (Dtype)0., std_angle_, &curr_angle_);
				curr_angle_ = curr_angle_ > -180 ? curr_angle_ : -180;
				curr_angle_ = curr_angle_ < 180 ? curr_angle_ : 180;
				TMatFromParam(ROTATION, curr_angle_, curr_angle_, tmat_cpu_data);
			}
			if (need_scale_){
				caffe_rng_gaussian(1, (Dtype)1., std_scale_, &curr_scale_);
				//clip to be in [min_scale_, max_scale_]
				curr_scale_ = curr_scale_ > min_scale_ ? curr_scale_ : min_scale_;
				curr_scale_ = curr_scale_ < max_scale_ ? curr_scale_ : max_scale_;
				TMatFromParam(SCALE, curr_scale_, curr_scale_, tmat_cpu_data);
			}
			if (need_shift_){
				Dtype shift_std_x = std_dx_prop_ * Width_;
				Dtype shift_std_y = std_dy_prop_ * Height_;
				caffe_rng_gaussian(1, (Dtype)0., shift_std_x, &curr_shift_x_);
				caffe_rng_gaussian(1, (Dtype)0., shift_std_y, &curr_shift_y_);
				Dtype max_shift_pixels_width = max_shift_prop_ * Width_;
				Dtype max_shift_pixels_height = max_shift_prop_ * Height_;
				//clip shift proportion to be less or equal max_shift_prop_
				curr_shift_x_ = curr_shift_x_ < max_shift_pixels_width ? curr_shift_x_ : max_shift_pixels_width;
				curr_shift_x_ = curr_shift_x_ > (-max_shift_pixels_width) ? curr_shift_x_ : (-max_shift_pixels_width);
				curr_shift_y_ = curr_shift_y_ < max_shift_pixels_height ? curr_shift_y_ : max_shift_pixels_height;
				curr_shift_y_ = curr_shift_y_ > (-max_shift_pixels_height) ? curr_shift_y_ : (-max_shift_pixels_height);
				TMatFromParam(SHIFT, curr_shift_x_, curr_shift_y_, tmat_cpu_data);
			}
			break;
		default:
			LOG(FATAL) << "Unkown sampling type";
			break;
		}
		//Canoincal size is set, so after finding the transformation,
		//crop or pad to that canonical size.
		//First find the coordinate matrix for this transformation
		//here we don't change the shape of the input 2D map
		GenCoordMatCrop_gpu(tmat_, Height_, Width_, original_coord_, coord_idx_, BORDER_, INTERP_);
	}

	template <typename Dtype>
	void RandomTransformLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*> &top){
		const int count = bottom[0]->count();
		const Dtype* bottom_data = bottom[0]->gpu_data();
		Dtype* top_data = top[0]->mutable_gpu_data();
		//randomly determine if we need to apply transformations or not
		//randomly decide to apply transformations
		if (needs_rand_){
			need_scale_ = scale_ && (Rand(2) == 1);
			need_rotation_ = rotation_ && (Rand(2) == 1);
			need_shift_ = shift_ && (Rand(2) == 1);
			//0.5 probability to apply transformations
			if ((scale_ + shift_ + rotation_) > 1){
				rand_ = Rand(2);
			}
			else{
				rand_ = 1;
			}
		}
		bool not_need_transform = (!need_shift_ && !need_scale_ && !need_rotation_) 
			|| this->phase_ == TEST || (needs_rand_ && rand_ == 0);
		//if there are no random transformations, we just copy bottom data to top blob
		//in test phase, we don't do any transformations
		if (not_need_transform){
			caffe_copy(count, bottom_data, top_data);
		}
		else{
			//get coordinate map matrix
			GetTransCoord_gpu();
			//Apply Imterpolation on bottom_data using tmat_[i] into top_data.
			InterpImageNN_gpu(bottom[0], coord_idx_.gpu_data(), top[0], INTERP_);
		}
	}

	template <typename Dtype>
	void RandomTransformLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*> &bottom){
		const int count = top[0]->count();
		const Dtype* top_diff = top[0]->gpu_diff();
		Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
		bool not_need_transform = (!need_shift_ && !need_scale_ && !need_rotation_)
			||(needs_rand_ && rand_ == 0);
		//Reset bottom diff.
		caffe_gpu_set(count, Dtype(0.), bottom_diff);
		if (propagate_down[0]){
			if (not_need_transform){
				caffe_copy(count, top_diff, bottom_diff);
			}
			else{
				BackPropagateErrorNN_gpu(top[0], coord_idx_.gpu_data(), bottom[0], INTERP_);
			}
		}
	}

	//since the atomicAdd gpu function in transform only support float,
	//so we only register float functions here
	INSTANTIATE_LAYER_GPU_FUNCS_FLOAT_ONLY(RandomTransformLayer);
}
