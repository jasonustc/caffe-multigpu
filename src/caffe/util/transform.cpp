/********************************************************************************
** Copyright(c) 2015 USTC Reserved.
** auth: Xu Shen
** mail: shenxu@mail.ustc.edu.cn
** date: 2015/11/4
** desc: Image Transformation Functions(CPU)
** Part of the code is borrowed from https://github.com/akanazawa/si-convnet
** @misc{kanazawa14,
** 	author    = {Angjoo Kanazawa and Abhishek Sharma and David W. Jacobs},
** 	title     = {Locally Scale-Invariant Convolutional Neural Networks},
** 	year      = {2014},
** 	url       = {http://arxiv.org/abs/1412.5104},
** 	Eprint = {arXiv:1412.5104}
** }
** Thanks for their nice work.
*********************************************************************************/
#include <cstdio>
#include <cmath>
#include <algorithm>
#include "caffe/util/transform.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/blob.hpp"

using std::min;
using std::max;

namespace caffe{

	void TMatFromRandom(float* tmat, RandType randType, float param1, float param2){
		std::fill(tmat, tmat + 9, 0);
		switch (randType){
		case caffe::GAUSSIAN:
			caffe_rng_gaussian<float>(9, param1, param2, tmat);
			break;
		case caffe::UNIFORM:
			caffe_rng_uniform<float>(9, param1, param2, tmat);
			break;
		default:
			LOG(FATAL) << "Unkown random type";
			break;
		}
		//only add noise to identity matrix
		tmat[0] += 1;
		tmat[4] += 1;
		tmat[8] = 1;
		tmat[2] = tmat[5] = 0;
	}

	/*
	 *Compute the transformation parameters in transform matrix
	 * 1) trans_type == ROTATION, rotation angle is stored in param1
	 * 2) trans_type == SCALE, scale proportion is stored in param1
	 * 3) trans_type == SHIFT, shift_x and shift_y(pixel) is stored in param1, param2, respectively.
	 * the last transform parameter in tmat will be updated, not reset: tmat_new = transform(tmat_old)
	 */
	void TMatFromParam(TransType transType, const float param1, const float param2, float *tmat, bool invert){
		// NOTE: if we want recursively apply mutiple transformations
		// we can not initialize here, because the previous transform parameters
		// will be flushed away.
		//initialize to identity
//		std::fill(tmat, tmat + 9, 0);
//		tmat[0] = tmat[4] = tmat[8] = 1;
		switch (transType)
		{
		case caffe::ROTATION:
			if (invert){
				AddRotation(-param1, tmat);
			}
			else{
				AddRotation(param1, tmat);
			}
			break;
		case caffe::SCALE:
			CHECK(param1 > 0) << "Scale has to be > 0: " << param1;
			if (invert){
				AddScale(1. / param1, tmat);
			}
			else{
				AddScale(param1, tmat);
			}
			break;
		case caffe::SHIFT:
			if (param1 != 0 || param2 != 0){
				if (invert){
					AddShift(-param1, -param2, tmat);
				}
				else{
					AddShift(param1, param2, tmat);
				}
			}
			break;
		default:
			LOG(FATAL) << "Unkown transform type";
			break;
		}
	}

	void AddScale(const float &scale, float *mat, const Direction dir){
		float tmp[9] = { scale, 0, 0, 0, scale, 0, 0, 0, 1 };
		AddTransform(mat, tmp, dir);
	}


	void AddRotation(const float &angle, float *mat, const Direction dir){
		//Angle in degrees
		float rad = angle * PI_F / 180;
		//static memory, system will release automatically
		float tmp[9] = { cos(rad), sin(rad), 0, -sin(rad), cos(rad), 0, 0, 0, 1 };
		AddTransform(mat, tmp, dir);
	}

	void AddShift(const float &dy, const float &dx, float *mat, const Direction dir){
		//dx is width, dy is height
		float tmp[9] = { 1, 0, 0, 0, 1, 0, dy, dx, 1};
		AddTransform(mat, tmp, dir);
	}

	/*
	 *all the 2D transformations can be modeled by the combination of following 3 basic
	 *transformations
	 *rotation:
	 *                   [cos\theta   sin\theta  0]
	 *[y' x' 1] = [y x 1][-sin\theta  cos\theta  0]
	 *                   [ 0            0        1]
	 *shift:
	 *                   [1  0  0]
	 *[y' x' 1] = [y x 1][0  1  0] 
	 *                   [dy dx 1]
	 *scale:
	 *                   [s_y  0    0]
	 *[y' x' 1] = [y x 1][0    s_x  0]
	 *                   [0    0    1]
	 *T_new = T_old * T_k
	 * X' = XT => X = X'T^{-1}
	 */
	void AddTransform(float *A, const float *B, const Direction dir){
		//matrix multiply A and B and store to A
		//i.e. A = A_copy * B + 0 * A
		//but gemm can't be done in inplace, so A has to be a copy of A
		//if dir == LEFT, does A = B * A_copy
		float A_copy[9];
		caffe_copy<float>(9, A, A_copy);
		dir == RIGHT ? caffe_cpu_gemm<float>(CblasNoTrans, CblasNoTrans, 3, 3, 3, 1.f,
			A_copy, B, 0.f, A) :
			caffe_cpu_gemm<float>(CblasNoTrans, CblasNoTrans, 3, 3, 3, 1.f,
			B, A_copy, 0.f, A);
	}

	//Following the inverse rule of 3x3 matrices using determinats
	//rewrites tmat into its inverse
	//TODO: convert tmat to double
	//use LU from lapack/blas if this is too numerically unstable?
	//invert index of images, image_transformed -> original_image
	void Invert3x3(float *A){
		float inv[9];
		//|A| = aei + bfg + cdh - (ceg + bdi + afh)
		//the determinant of matrix A
		/*A = [0 1 2]
		 *    [3 4 5]
		 *    [6 7 8]
		 *A^T = [0 3 6]      [+ - +]
		 *      [1 4 7] dot  [- + -]
		 *      [2 5 8]      [+ - +]
		 */
		float d1 = A[0] * A[4] * A[8] + A[1] * A[5] * A[6] + A[2] * A[3] * A[7];
		float d2 = A[0] * A[5] * A[7] + A[2] * A[4] * A[6] + A[8] * A[1] * A[3];
		float det = d1 - d2;
		CHECK_NE(det, 0);
		inv[0] = (A[8] * A[4] - A[5] * A[7]) / det;
		inv[1] = (A[7] * A[2] - A[1] * A[8]) / det;
		inv[2] = (A[1] * A[5] - A[4] * A[2]) / det;
		inv[3] = (A[6] * A[5] - A[3] * A[8]) / det;
		inv[4] = (A[0] * A[8] - A[6] * A[2]) / det;
		inv[5] = (A[3] * A[2] - A[0] * A[5]) / det;
		inv[6] = (A[3] * A[7] - A[6] * A[4]) / det;
		inv[7] = (A[6] * A[1] - A[0] * A[7]) / det;
		inv[8] = (A[0] * A[4] - A[3] * A[1]) / det;
		caffe_copy(9, inv, A);
	}

	//get the reflect location in matrix
	//just mirror pixels to the border
	template <typename Dtype> void Reflect(Dtype &val, const int size){
		if (val < 0.){
			val = -floor(val);
			val = static_cast<Dtype>(static_cast<int>(val) % (2 * size - 2));
		}
		if (val >= size){
			val = 2 * size - 2 - val;
		}
	}
	template void Reflect<float>(float &val, const int size);

	//if out of range, just repeat the pixels on the bord
	template <typename Dtype> void Clamp(Dtype &val, const int size){
		val = max(static_cast<Dtype>(0.), min(val, static_cast<Dtype>(size - 1)));
	}

	template void Clamp<float>(float &val, const int size);


	void generate_nn_coord(const int &height, const int &width,
		const int &height_new, const int &width_new,
		const Border &border, const float* coord_data_res,
		float* &coord_data){
		float old_cy = static_cast<float>(height - 1) / 2.;
		float old_cx = static_cast<float>(width - 1) / 2.;
		//copy over results over after applying reflection dropping the 3rd dim
		//also need to add the old_center
		for (int ind = 0; ind < height_new * width_new; ++ind){
			float row = round(coord_data_res[3 * ind] + old_cy);
			float col = round(coord_data_res[3 * ind + 1] + old_cx);
			switch (border)
			{
			case CROP:
				if ((row >= height || row < 0) || (col >= width) || (col < 0)){
					coord_data[ind] = -1;
					//why continue here?
					continue;
				}
				break;
			case CLAMP:
				Clamp(row, height);
				Clamp(col, width);
				break;
			case REFLECT:
				Reflect(row, height);
				Reflect(col, width);
				break;
			default:
				LOG(FATAL) << "Unkown border mode.";
				break;
			}
			//save index
			coord_data[ind] = round(row) * width + round(col);
		}
	}

	/*
	 *Going to save [ind(p00), ind(p11), dc(x-x0), dr(y-y0)] for each pixel 
	 *in the new image
	 *These will be stored in 4xHxW matrix read in columnwise manner i.e.
	 *coord = [ind_0(p00), ind_1(p00),...
	 *         ind_0(p11), ind_1(p11),...
	 *         dc_0, dc_1, ...
	 *         dr_0, dr_1, ...]
	 * bc so that the first row is the NN-coord
	 *& compatible with max-pooling range checking later(see MaxTransSetSwitch).
	 * [(trunc(x),trunc(y)				  (trunc(x+1), trunc(y)) ]
	 * [                     (row, col)                          ]
	 * [(trunc(y+1), trunc(x))            (trunc(x+1), trunc(y+1)]
	 *
	 */
	void generate_bilinear_coord(const int &height, const int &width,
		const int &height_new, const int &width_new,
		const Border &border, const float* coord_data_res,
		float* &coord_data){
		float old_cy = static_cast<float>(height - 1) / 2.;
		float old_cx = static_cast<float>(width - 1) / 2.;

		//copy over results over after applying reflection dropping the 3rd
		//also need to add the old_center
		//first remove the influence of the center, then resume to the old center
		float row, col, row0, col0, row1, col1, dc, dr;
		int N = height_new * width_new;
		for (int ind = 0; ind < N; ++ind){
			//add center
			row = coord_data_res[3 * ind] + old_cy;
			col = coord_data_res[3 * ind + 1] + old_cx;
			//p00=>(r0, c0) p11=>(r1, c1)
			//save index
			switch (border)
			{
			case CROP:
				//exceed half pixel
				//ignored in bilinear interpolation
				if ((row >= height - 0.5 || row < -0.5) || 
					(col >= width - 0.5 || col < -0.5)){
					coord_data[ind] = -1;
					continue;
				}
				break;
			case CLAMP:
				Clamp(row, height);
				Clamp(row, height);
				break;
			case REFLECT:
				Reflect(row, height);
				Reflect(col, width);
				break;
			default:
				LOG(FATAL) << "Unknown border mode.";
				break;
			}
			//p00
			//round towards zero, the magnitude should not larger than x.
			row0 = trunc(row);
			col0 = trunc(col);
			//p11
			row1 = trunc(row + 1) > (height - 1) ? height - 1 : trunc(row + 1);
			col1 = trunc(col + 1) > (width - 1) ? width - 1 : trunc(col + 1);

			//if p00 is outside, don't compute difference
			//difference of truncation
			//if no difference or difference is an int, we don't need to do 
			//bilinear interpolation
			dc = col1 == col0 ? 0 : col - col0;
			dr = row1 == row0 ? 0 : row - row0;
			DCHECK(dc >= 0) << "dc has to be pos " << dc;
			DCHECK(dr >= 0) << "dr has to be pos " << dr;
			//left up point
			coord_data[ind] = row0 * width + col0;
			//right down point
			coord_data[ind + N] = row1 * width + col1;
			//column difference
			coord_data[ind + 2 * N] = dc;
			//row difference
			coord_data[ind + 3 * N] = dr;
		}
	}

	//This doesn't change the size of the input
	void GenCoordMatCrop_cpu(Blob<float>& tmat,
		const int &height, const int &width, Blob<float>& ori_coord,
		Blob<float>& coord_idx, const Border &border, const Interp &interp){

		float* tmat_data = tmat.mutable_cpu_data();

		//transform tmat to it's inversed matrix
		Invert3x3(tmat_data);

		float cy = static_cast<float>(height - 1) / 2.;
		float cx = static_cast<float>(width - 1) / 2.;

		//substract center
		AddShift(-cy, -cx, tmat_data, LEFT);

		//we can use coord data and diff for buffer of coordinate
		//data, since it is only used after this computation
		float *coord_data_tmp = ori_coord.mutable_cpu_data();
		float *coord_data_res = ori_coord.mutable_cpu_diff();
		//put the final coordinates in coord_idx
		float *coord_data_final = coord_idx.mutable_cpu_data();

		//Apply transformation
		caffe_cpu_gemm<float>(CblasNoTrans, CblasNoTrans, height * width, 3, 3, 1.f,
			coord_data_tmp, tmat_data, 0.f, coord_data_res);

		//we can now put the final coordinate data into coord data again
		switch (interp)
		{
		case NN:
			generate_nn_coord(height, width, height, width, border, coord_data_res,
				coord_data_final);
			break;
		case BILINEAR:
			generate_bilinear_coord(height, width, height, width, border,
				coord_data_res, coord_data_final);
			break;
		default:
			LOG(ERROR) << "Unkown interpolation mode " << interp;
			break;
		}
	}

	//fills width * height by 3 matrix that holds identity homogeneous coordinates
	//get mat index
	void GenBasicCoordMat(float* coord, const int &width, const int &height){
		int row, col;
		for (int ind = 0; ind < width * height; ++ind){
			//compute subscripts from this index
			row = ind / width;
			col = ind % width;

			//(y, x, 1)
			coord[3 * ind] = row;
			coord[3 * ind + 1] = col;
			coord[3 * ind + 2] = 1;
		}
	}

	template <typename Dtype>
	void InterpImageNN_cpu(const Blob<Dtype>* orig, const float* coord,
		Blob<Dtype>* warped, const Interp &interp){
		switch (interp)
		{
		case NN:
			nn_interpolation(orig, coord, warped);
			break;
		case BILINEAR:
			bilinear_interpolation(orig, coord, warped);
			break;
		default:
			LOG(ERROR) << "Unkown interpolation mode" << interp;
			break;
		}
	}

	template void InterpImageNN_cpu(const Blob<float>* orig, const float* coord,
		Blob<float>* warped, const Interp &interp);
//	template void InterpImageNN_cpu(const Blob<double>* orig, const float* coord,
//		Blob<double>* warped, const Interp &interp);

	//just like crop
	template <typename Dtype>
	void nn_interpolation(const Blob<Dtype>* &orig, const float* &coord,
		Blob<Dtype>* &warped){
		//Get the parameters from the original and warped and apply the
		//transformation to it.
		int ind_warped, ind_orig, h_orig, w_orig;
		int width_orig = orig->width();
		int height_orig = orig->height();
		int num = warped->num();
		int channels = warped->channels();
		int height = warped->height();
		int width = warped->width();

		const Dtype *orig_data = orig->cpu_data();
		Dtype* warped_data = warped->mutable_cpu_data();

		for (int n = 0; n < num; ++n){//for each img
			for (int c = 0; c < channels; ++c){//for each channel
				for (int h = 0; h < height; ++h){
					for (int w = 0; w < width; ++w){
						ind_warped = h * width + w;//index in warped image
						ind_orig = static_cast<int>(coord[ind_warped]);
						if (ind_orig >= 0){ //do only if valid index
							h_orig = ind_orig / width_orig; //row in original
							w_orig = ind_orig % width_orig; //col in original
							warped_data[((n * channels + c) * height + h) * width + w] =
								orig_data[((n * channels + c) * height_orig + h_orig) * width_orig + w_orig];
						}
						else{
							warped_data[((n*channels + c) * height + h) * width + w] = 0;
						}
					}
				}
			}
		}
	}

	template <typename Dtype>
	void bilinear_interpolation(const Blob<Dtype>* orig, const float* coord,
		Blob<Dtype>* warped){
		//Get the parameters from the original and warped and apply the
		//transformation to it.
		int ind_warped, ind_orig, r0, c0, r1, c1, ind_p11;
		float dc, dr, w00, w01, w10, w11;
		float p00, p01, p10, p11;
		int width_orig = orig->width();
		int height_orig = orig->height();
		int num = warped->num();
		int channels = warped->channels();
		int height = warped->height();
		int width = warped->width();
		int N = width * height;
		const Dtype *orig_data = orig->cpu_data();
		Dtype* warped_data = warped->mutable_cpu_data();

		for (int n = 0; n < num; ++n){//for each image
			for (int c = 0; c < channels; ++c){//for each channel
				for (int h = 0; h < height; ++h){
					for (int w = 0; w < width; ++w){
						ind_warped = h * width + w;//index for coordinate in new image
						//because we use the inverse tmat to get coord
						//so we can infer original coord(x_orig, y_orig) by (x,y)
						//in the new image and coord
						ind_orig = static_cast<int>(coord[ind_warped]);//p00
						if (ind_orig >= 0){//do only if p00 is valid index
							r0 = ind_orig / width_orig;
							c0 = ind_orig % width_orig;
							//Coordinates are stored as 4 x N matrix
							ind_p11 = static_cast<int>(coord[ind_warped + N]);
							r1 = ind_p11 / width;
							c1 = ind_p11 % width;

							dc = coord[ind_warped + 2 * N];
							dr = coord[ind_warped + 3 * N];

							//bilinear interpolation
							//f(x,y) \approx f(0,0)(1-x)(1-y) + f(1,0)x(1-y) + f(0,1)(1-x)y + f(1,1)xy
							//in the little one pixel square, dr and dc just corresponding to y and x
							w00 = (1 - dc) * (1 - dr);
							w01 = (1 - dr) * dc;
							w10 = (1 - dc) * dr;
							w11 = dr * dc;

							int offset = (n * channels + c) * height_orig;
							p00 = orig_data[(offset + r0) * width_orig + c0];
							p01 = orig_data[(offset + r0) * width_orig + c1];
							p10 = orig_data[(offset + r1) * width_orig + c0];
							p11 = orig_data[(offset + r1) * width_orig + c1];

							warped_data[((n*channels + c) * height + h) * width + w] =
								w00 * p00 + w01 * p01 + w10 * p10 + w11 * p11;
						}
						else{
							warped_data[((n*channels + c) * height + h) * width + w] = 0;
						}
					}
				}
			}
		}
	}
	template void bilinear_interpolation(const Blob<float>* orig, const float* coord,
		Blob<float>* warped);

	//back propagation according to the corresponding indexes
	//in propatation
	template <typename Dtype>
	void BackPropagateErrorNN_cpu(const Blob<Dtype>* top, const float* coord,
		Blob<Dtype>* bottom, const Interp &interp){
		switch (interp)
		{
		case NN:
			nn_backpropagation(top, coord, bottom);
			break;
		case BILINEAR:
			bilinear_backpropagation(top, coord, bottom);
			break;
		default:
			LOG(ERROR) << "Unkown interpolation mode " << interp;
			break;
		}
	}

	//Explicit instantiation
	template void BackPropagateErrorNN_cpu(const Blob<float>* top, const float* coord,
		Blob<float>* bottom, const Interp &interp);
//	template void PropagateErrorNN_cpu(const Blob<double>* top, const float* coord,
//		Blob<double>* bottom, const Interp &interp);

	template <typename Dtype>
	void nn_backpropagation(const Blob<Dtype>* & top, const float* &coord,
		Blob<Dtype>* &bottom){
		//I will simply take the error at each location in top and add it to the 
		//corresponding neuron
		//in the bottom blob based on the coord indices
		//IMP: IT IS ASSUMED THAT THE BOTTOM DIFF IS PROPERLY PRE-INITIALIZED
		//I.E. HAS ALL ZEROS OR PROPER ERROR VALUES
		int ind_top, ind_bottom, h_bottom, w_bottom;

		int num = top->num();
		int channels = top->channels();
		int height = top->height();
		int width = top->width();

		int height_bottom = bottom->height();
		int width_bottom = bottom->width();

		const Dtype *top_diff = top->cpu_diff();
		Dtype* bottom_diff = bottom->mutable_cpu_diff();

		//Loop over top locations
		for (int n = 0; n < num; ++n){
			for (int c = 0; c < channels; ++c){
				for (int h = 0; h < height; ++h){
					for (int w = 0; w < width; ++w){
						ind_top = h*width + w;
						ind_bottom = static_cast<int>(coord[ind_top]);
						if (ind_bottom >= 0){//do only for valid index
							h_bottom = ind_bottom / width_bottom; //row
							w_bottom = ind_bottom % width_bottom; //col
							bottom_diff[((n* channels + c) * height_bottom + h_bottom) * width_bottom
								+ w_bottom] +=
								top_diff[((n*channels + c) * height + h) * width + w];
						}
					}
				}
			}
		}
	}

	template <typename Dtype>
	void bilinear_backpropagation(const Blob<Dtype>* & top, const float* & coord,
		Blob<Dtype>* &bottom){
		//AJ: Just like forward image interpolation
		//IMP: IT IS ASSUMED THAT THE BOTTOM IS PROPERLY PRE_INITIALIZED AND HAS ALL
		//ZEROS OR PROPER ERROR VALUES
		int ind_top, ind_bottom, r0, c0, ind_p11, r1, c1;
		float dc, dr, w00, w01, w10, w11;
		int num = top->num();
		int channels = top->channels();
		int height = top->height();
		int width = top->width();

		int N = width* height;

		int height_bottom = bottom->height();
		int width_bottom = bottom->width();

		const Dtype *top_diff = top->cpu_diff();
		Dtype* bottom_diff = bottom->mutable_cpu_diff();

		//Loop over top locations
		for (int n = 0; n < num; ++n){
			for (int c = 0; c < channels; ++c){
				for (int h = 0; h < height; ++h){
					for (int w = 0; w < width; ++w){
						ind_top = h * width + w;
						ind_bottom = static_cast<int>(coord[ind_top]);
						if (ind_bottom >= 0){ //do only if valid index
							r0 = ind_bottom / width_bottom; //row
							c0 = ind_bottom % width_bottom; //col

							//Coordinates are stores as 4 x N matrix
							ind_p11 = static_cast<int>(coord[ind_top + N]);
							r1 = ind_p11 / width_bottom;
							c1 = ind_p11 % width_bottom;

							dc = coord[ind_top + 2 * N];
							dr = coord[ind_top + 3 * N];

							w00 = (1 - dc)*(1 - dr);
							w01 = (1 - dr) * dc;
							w10 = (1 - dc) * dr;
							w11 = dr * dc;

							int offset = (n * channels + c) * height_bottom;

							float top_error =
								top_diff[((n * channels + c) * height + h) * width + w];

							//propagate error after weighting with its bilinear coefficients
							//p00
							bottom_diff[(offset + r0) * width_bottom + c0] += w00 * top_error;
							//p01
							bottom_diff[(offset + r0) * width_bottom + c1] += w01 * top_error;
							//p10
							bottom_diff[(offset + r1) * width_bottom + c0] += w10 * top_error;
							//p11
							bottom_diff[(offset + r1) * width_bottom + c1] += w11 * top_error;
						}//end if index if valid
					}
				}
			}
		}
	}
}