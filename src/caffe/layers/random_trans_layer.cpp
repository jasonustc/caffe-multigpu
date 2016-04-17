/********************************************************************************
** Copyright(c) 2015 USTC Reserved.
** auth: Xu Shen
** mail: shenxu@mail.ustc.edu.cn
** date: 2015/11/4
** desc: RandomTransformLayer(CPU)
*********************************************************************************/
#include <vector>
#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/filler.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/random_transform_layer.hpp"

namespace caffe{
	template <typename Dtype>
	void RandomTransformLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top){
		Layer<Dtype>::LayerSetUp(bottom, top);
		sample_type_ = this->layer_param_.rand_trans_param().sample_type();
		CHECK_EQ(bottom.size(), 1) << "RandomTranform Layer only takes one single blob as input.";
		CHECK_EQ(top.size(), 1) << "RandomTransform Layer only takes one single blob as output.";
		LOG(INFO) << "Random Transform Layer using border type: " << this->layer_param_.rand_trans_param().border()
			<< ", using interpolation: " << this->layer_param_.rand_trans_param().interp();
		switch (sample_type_){
		case RandTransformParameter_SampleType_UNIFORM:
			rotation_ = this->layer_param_.rand_trans_param().has_start_angle() &&
				this->layer_param_.rand_trans_param().has_end_angle();
			shift_ = this->layer_param_.rand_trans_param().has_dx_prop() &&
				this->layer_param_.rand_trans_param().has_dy_prop();
			scale_ = this->layer_param_.rand_trans_param().has_start_scale() &&
				this->layer_param_.rand_trans_param().has_end_scale();

			if (rotation_){
				start_angle_ = this->layer_param_.rand_trans_param().start_angle();
				end_angle_ = this->layer_param_.rand_trans_param().end_angle();
				CHECK_GE(start_angle_, -180) << "start angle should be larger than -180";
				CHECK_LE(end_angle_, 180) << "end angle should be less than 180";
				CHECK_LE(start_angle_, end_angle_);
				LOG(INFO) << "random rotate in [" << start_angle_ << "," << end_angle_ << "].";
			}
			if (shift_){
				dx_prop_ = this->layer_param_.rand_trans_param().dx_prop();
				dy_prop_ = this->layer_param_.rand_trans_param().dy_prop();
				CHECK_GE(dx_prop_, 0);
				CHECK_LE(dx_prop_, 1);
				CHECK_GE(dy_prop_, 0);
				CHECK_LE(dy_prop_, 1);
				LOG(INFO) << "Random shift image by dx <= " << dx_prop_ <<
					"*Width_, dy <= " << dy_prop_ << "*Height_.";
			}
			if (scale_){
				start_scale_ = this->layer_param_.rand_trans_param().start_scale();
				end_scale_ = this->layer_param_.rand_trans_param().end_scale();
				CHECK_GT(start_scale_, 0);
				CHECK_LE(start_scale_, end_scale_);
				LOG(INFO) << "Random scale image in [" << start_scale_ << "," << end_scale_ << "]";
			}
			break;
		case RandTransformParameter_SampleType_GAUSSIAN:
			rotation_ = this->layer_param_.rand_trans_param().has_std_angle();
			scale_ = this->layer_param_.rand_trans_param().has_std_scale();
			shift_ = this->layer_param_.rand_trans_param().has_std_dx_prop() &&
				this->layer_param_.rand_trans_param().has_std_dy_prop();
			if (scale_){
				std_scale_ = this->layer_param_.rand_trans_param().std_scale();
				CHECK_GT(std_scale_, 0) << "std of scale sampling should be positive";
				LOG(INFO) << "random scale input by variance of " << std_scale_;
			}
			if (rotation_){
				std_angle_ = this->layer_param_.rand_trans_param().std_angle();
				CHECK_GT(std_angle_, 0) << "std of angle sampling should be positive";
				LOG(INFO) << "random rotate input by variance of " << std_angle_;
			}
			if (shift_){
				std_dx_prop_ = this->layer_param_.rand_trans_param().std_dx_prop();
				std_dy_prop_ = this->layer_param_.rand_trans_param().std_dy_prop();
				CHECK_GT(std_dx_prop_, 0) << "std of shift proportion should be positive";
				CHECK_LE(std_dx_prop_, 1) << "std of shift proportions should be less than 1";
				CHECK_GT(std_dy_prop_, 0);
				CHECK_LE(std_dy_prop_, 1);
				LOG(INFO) << "random shift input by variance of x,y " 
					<< std_dx_prop_ << "," << std_dy_prop_;
			}
			min_scale_ = this->layer_param_.rand_trans_param().min_scale();
			max_scale_ = this->layer_param_.rand_trans_param().max_scale();
			max_shift_prop_ = this->layer_param_.rand_trans_param().max_shift_prop();
			break;
		default:
			LOG(FATAL) << "Unkown sampling type";
			break;
		}
		Height_ = bottom[0]->height();
		Width_ = bottom[0]->width();
		BORDER_ = static_cast<Border>(this->layer_param_.rand_trans_param().border());
		INTERP_ = static_cast<Interp>(this->layer_param_.rand_trans_param().interp());
		InitRand();
		need_scale_ = scale_;
		need_rotation_ = rotation_;
		need_shift_ = shift_;
	}

	template <typename Dtype>
	void RandomTransformLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top){
		top[0]->ReshapeLike(*bottom[0]);
		switch (INTERP_){
		case NN:
			coord_idx_.Reshape(1, 1, Height_ * Width_, 1);
			break;
		case BILINEAR:
			coord_idx_.Reshape(1, 1, Height_ * Width_ * 4, 1);
			break;
		default:
			LOG(FATAL) << "Unkown pooling method.";
		}
		//to store the original row and colum index data of the matrix
		original_coord_.Reshape(1, 1, Height_ * Width_ * 3, 1);
		//the order stored in the original_coord_ is (y0, x0, 1, y1, x1, 1, ...)
		GenBasicCoordMat(original_coord_.mutable_cpu_data(), Width_, Height_);
		tmat_.Reshape(1, 1, 3, 3);
	}

	template <typename Dtype>
	void RandomTransformLayer<Dtype>::GetTransCoord_cpu(){
		InitTransform();
		float* tmat_data = tmat_.mutable_cpu_data();
		//compute transformation matrix
		switch (sample_type_){
		case RandTransformParameter_SampleType_UNIFORM:
			if (need_rotation_){
				//randomly generate rotation angle
				caffe_rng_uniform(1, start_angle_, end_angle_, &curr_angle_);
				TMatFromParam(ROTATION, curr_angle_, curr_angle_, tmat_data);
			}
			if (need_scale_){
				caffe_rng_uniform(1, start_scale_, end_scale_, &curr_scale_);
				TMatFromParam(SCALE, curr_scale_, curr_scale_, tmat_data);
			}
			if (need_shift_){
				float shift_pixels_x = dx_prop_ * Width_;
				float shift_pixels_y = dy_prop_ * Height_;
				caffe_rng_uniform(1, -shift_pixels_x, shift_pixels_x, &curr_shift_x_);
				caffe_rng_uniform(1, -shift_pixels_y, shift_pixels_y, &curr_shift_y_);
				TMatFromParam(SHIFT, curr_shift_x_, curr_shift_y_, tmat_data);
			}
			break;
		case RandTransformParameter_SampleType_GAUSSIAN:
			if (need_rotation_){
				//clip to be in [-180, 180]
				caffe_rng_gaussian(1, (Dtype)0., std_angle_, &curr_angle_);
				curr_angle_ = curr_angle_ > -180 ? curr_angle_ : -180;
				curr_angle_ = curr_angle_ < 180 ? curr_angle_ : 180;
				TMatFromParam(ROTATION, curr_angle_, curr_angle_, tmat_data);
			}
			if (need_scale_){
				caffe_rng_gaussian(1, (Dtype)1., std_scale_, &curr_scale_);
				//clip to be in [min_scale_, max_scale_]
				curr_scale_ = curr_scale_ > min_scale_ ? curr_scale_ : min_scale_;
				curr_scale_ = curr_scale_ < max_scale_ ? curr_scale_ : max_scale_;
				TMatFromParam(SCALE, curr_scale_, curr_scale_, tmat_data);
			}
			if (need_shift_){
				Dtype shift_std_x = std_dx_prop_ * Width_;
				Dtype shift_std_y = std_dy_prop_ * Height_;
				caffe_rng_gaussian(1, (Dtype)0., shift_std_x, &curr_shift_x_);
				caffe_rng_gaussian(1, (Dtype)0., shift_std_y, &curr_shift_y_);
				Dtype max_shift_pixels_width = max_shift_prop_ * Width_;
				Dtype max_shift_pixels_height = max_shift_prop_ * Height_;
				//clip shift proportion to less or equal max_shift_prop_
				curr_shift_x_ = curr_shift_x_ < max_shift_pixels_width ? curr_shift_x_ : max_shift_pixels_width;
				curr_shift_x_ = curr_shift_x_ > (-max_shift_pixels_width) ? curr_shift_x_ : (-max_shift_pixels_width);
				curr_shift_y_ = curr_shift_y_ < max_shift_pixels_height ? curr_shift_y_ : max_shift_pixels_height;
				curr_shift_y_ = curr_shift_y_ > (-max_shift_pixels_height) ? curr_shift_y_ : (-max_shift_pixels_height);
				TMatFromParam(SHIFT, curr_shift_x_, curr_shift_y_, tmat_data);
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
		//wo we don't need crop operation here
		GenCoordMatCrop_cpu(tmat_, Height_, Width_, original_coord_, coord_idx_, BORDER_, INTERP_);
	}

	template <typename Dtype>
	void RandomTransformLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top){
		const int count = bottom[0]->count();
		const Dtype* bottom_data = bottom[0]->cpu_data();
		Dtype* top_data = top[0]->mutable_cpu_data();
		const int channels = bottom[0]->channels();
		//randomly decide to apply transformations
		if (needs_rand_){
			need_scale_ = scale_ && (Rand(2) == 1);
			need_rotation_ = rotation_ && (Rand(2) == 1);
			need_shift_ = shift_ && (Rand(2) == 1);
            // keep 0.5 probability to apply transformations
            if ((scale_ + rotation_ + shift_) > 1){
                rand_ = Rand(2);
            }
            else{
                rand_ = 1;
            }
		}
		bool not_need_transform = (!need_shift_ && !need_scale_ && !need_rotation_) 
			|| this->phase_ == TEST || (needs_rand_ && rand_ == 0);
		//if there are no random transformations, we just copy bottom data to top blob
		//in test phase, we just don't do any transformations
		if (not_need_transform){
			caffe_copy(count, bottom_data, top_data);
		}
		else{
			GetTransCoord_cpu();
			//apply Interpolation on bottom_data using tmat_[i] into top_data
			//the coord_idx_[i] will be of size as the output data
			InterpImageNN_cpu(bottom[0], coord_idx_.cpu_data(), top[0], INTERP_);
		}
	}

	//Backward has to return 1 bottom
	//Note that backwards coordinate indices are also stored in data
	template <typename Dtype>
	void RandomTransformLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom){
		//Reset bottom diff.
		const Dtype* top_diff = top[0]->cpu_diff();
		const int count = top[0]->count();
		CHECK_EQ(bottom[0]->count(), top[0]->count());
		Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
		bool not_need_transform = (!need_shift_ && !need_scale_ && !need_rotation_)
			||(needs_rand_ && rand_ == 0);
		//we must set all bottom diffs to zero before the backpropagation
		caffe_set(count, Dtype(0.), bottom_diff);
		if (propagate_down[0]){
			if (not_need_transform){
				caffe_copy(count, top_diff, bottom_diff);
			}
			else{
				BackPropagateErrorNN_cpu(top[0], coord_idx_.cpu_data(), bottom[0], INTERP_);
			}
		}
	}
#ifdef CPU_ONLY
	STUB_GPU(RandomTransformLayer);
#endif

	INSTANTIATE_CLASS_FLOAT_ONLY(RandomTransformLayer);

	//since the atomicAdd gpu function in transform only support float,
	//so we only register float functions here
	REGISTER_LAYER_CLASS_FLOAT_ONLY(RandomTransform);
}
