/********************************************************************************
** Copyright(c) 2015 USTC Reserved.
** auth: Xu Shen
** mail: shenxu@mail.ustc.edu.cn
** date: 2015/11/4
** desc: Image Transformation Functions(GPU)
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
#include <cstdlib>
#include <cstring>
#include <stdio.h>
#include <thrust/device_vector.h>
#include <thrust/count.h>

#include "caffe/common.hpp"
#include "caffe/util/transform.hpp"
#include "caffe/blob.hpp"

namespace caffe{
	template <typename Dtype>
	__device__ void Reflect_gpu(Dtype& val, const int size){
		if (val < Dtype(0)){
			val = -floor(val);
			val = static_cast<Dtype>(static_cast<int>(val) % (2 * size - 1));
		}
		if (val >= size){
			val = 2 * size - 2 - val;
		}
	}

	template <typename Dtype>
	__device__ void Clamp_gpu(Dtype& val, const int size){
		val = max(static_cast<Dtype>(0.), min(val, static_cast<Dtype>(size - 1)));
	}

	template <typename Dtype>
	__global__ void generate_nn_coord_kernel(const int N, const int height, const int width,
		const int height_new, const int width_new, const Border& border,
		const float* coord_data_res, float* &coord_data){
		float old_cy = static_cast<float>(height - 1) / 2.;
		float old_cx = static_cast<float>(width - 1) / 2.;
		CUDA_KERNEL_LOOP(index, N){
			//nearest pixel
			float row = round(coord_data_res[3 * index] + old_cy);
			float col = round(coord_data_res[3 * index + 1] + old_cx);
			switch (border)
			{
			case CROP:
				if ((row >= height || row < 0) || (col >= width) || (col < 0)){
					coord_data[index] = Dtype(-1);
					continue;
				}
				break;
			case CLAMP:
				Clamp_gpu(row, height);
				Clamp_gpu(col, width);
				break;
			case REFLECT:
				Reflect_gpu(row, height);
				Reflect_gpu(col, width);
				break;
			default:
				break;
			}
			coord_data[index] = round(row) * width + round(col);
		}
	}

	template <typename Dtype>
	__global__ void generate_bilinear_coord_kernel(const int N, const int height, const int width,
		const int height_new, const int width_new, const Border border,
		const float* coord_data_res, float* coord_data){
		CUDA_KERNEL_LOOP(index, N){
			float old_cy = static_cast<float>(height - 1) / 2.;
			float old_cx = static_cast<float>(width - 1) / 2.;
			float row = coord_data_res[3 * index] + old_cy;
			float col = coord_data_res[3 * index + 1] + old_cx;
			//p00 => (r0, c0) p11 => (r1,c1)
			switch (border)
			{
			case CROP:
				//skip interpolation
				if ((row >= height - 0.5 || row < -0.5) || (col >= width - 0.5) || 
					(col < -0.5)){
					coord_data[index] = Dtype(-1);
					continue;
				}
				break;
			case CLAMP:
				Clamp_gpu(row, height);
				Clamp_gpu(col, width);
				break;
			case REFLECT:
				Reflect_gpu(row, height);
				Reflect_gpu(col, width);
				break;
			default:
				break;
			}
			//p00, trunc(x), trunc(y)
			float row0 = trunc(row);
			float col0 = trunc(col);
			//p11
			float row1 = trunc(row + 1) > (height - 1) ? height - 1 : trunc(row + 1);
			float col1 = trunc(col + 1) > (width - 1) ? width - 1 : trunc(col + 1);

			//if p00 is outside, don't compute difference
			float dc = col0 == col1 ? 0 : col - col0;
			float dr = row0 == row1 ? 0 : row - row0;

			//left up point
			coord_data[index] = row0 * width + col0;
			//right down point
			coord_data[index + N] = row1 * width + col1;
			//column difference
			coord_data[index + 2 * N] = dc;
			//row difference
			coord_data[index + 3 * N] = dr;
		}
	}

	//currently, we only support float here
	void GenCoordMatCrop_gpu(Blob<float>& tmat, const int height, const int width,
		Blob<float>& ori_coord, Blob<float>& coord_idx, const Border& border, const Interp& interp){
		float* tmat_cpu_data = tmat.mutable_cpu_data();
		CHECK(border == CLAMP || border == CROP || border == REFLECT) << 
			"Unknown border type: " << border;
		//transform to inverse new_image => ori_image
		Invert3x3(tmat_cpu_data);

		float cy = static_cast<float>(height - 1) / 2.;
		float cx = static_cast<float>(width - 1) / 2.;

		//substract center
		AddShift(-cy, -cx, tmat_cpu_data, LEFT);

		//we can use ori_coord data and diff for buffer of coordinates
		//since it is only used in this step
		const float *coord_data_tmp = ori_coord.gpu_data();
		float *coord_data_res = ori_coord.mutable_gpu_diff();
		float *tmat_gpu_data = tmat.mutable_gpu_data();

		//Apply transformation
		caffe_gpu_gemm<float>(CblasNoTrans, CblasNoTrans, height * width, 3, 3, 1.f,
			coord_data_tmp, tmat_gpu_data, 0.f, coord_data_res);

		//save the final result into coord_idx
		float *coord_data_final = coord_idx.mutable_gpu_data();
		int n = height * width;
		switch (interp)
		{
		case NN:
			generate_nn_coord_kernel<float><< <CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS >>>(
				n, height, width, height, width, border, coord_data_res, coord_data_final);
			break;
		case BILINEAR:
			generate_bilinear_coord_kernel<float><< <CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS >>>(
				n, height, width, height, width, border, coord_data_res, coord_data_final);
			break;
		default:
			LOG(FATAL) << "Unkown interpolation type " << interp;
			break;
		}
		CUDA_POST_KERNEL_CHECK;
	}

	template <typename Dtype>
	__global__ void nn_interpolation_kernel(const int nthreads, const Dtype *oldDPtr,
		const int oldSheetCount, Dtype* newDPtr,
		const int newSheetCount, const float* coord){
		int index = blockIdx.x * blockDim.x + threadIdx.x;
		if (index < nthreads){
			int offset = index % newSheetCount;
			int numSheet = index / newSheetCount;
			int backSheetOffset = static_cast<int>(coord[offset]);
			if (backSheetOffset >= 0){
				newDPtr[numSheet * newSheetCount + offset] =
					oldDPtr[numSheet * oldSheetCount + backSheetOffset];
			}
			else{
				newDPtr[numSheet * newSheetCount + offset] = 0;
			}
		}
	}

	template <typename Dtype>
	__global__ void bilinear_interpolation_kernel(const int nthreads, const Dtype* oldDPtr,
		const int oldSheetCount, Dtype* newDPtr, const int newSheetCount, const float* coord,
		const int W){
		int index = blockIdx.x * blockDim.x + threadIdx.x;
		if (index < nthreads){
			int offset = index % newSheetCount; //p00: r0 * W + c0
			int numSheet = index / newSheetCount;
			int backSheetOffset = static_cast<int>(coord[offset]);
			if (backSheetOffset >= 0){
				int c0 = backSheetOffset % W;
				//p11: r1 * W + c1
				int ind_p11 = static_cast<int>(coord[offset + newSheetCount]);
				int c1 = ind_p11 % W;

				int ind_p01 = backSheetOffset - c0 + c1;//r0 * W + c1
				int ind_p10 = ind_p11 - c1 + c0; //r1 * W + c0

				float dc = coord[offset + 2 * newSheetCount];
				float dr = coord[offset + 3 * newSheetCount];

				float w00 = (1 - dc) * (1 - dr);
				float w01 = (1 - dr) * dc;
				float w10 = (1 - dc) * dr;
				float w11 = dr * dc;

				int bigOffset = numSheet * oldSheetCount;
				newDPtr[numSheet * newSheetCount + offset] =
					w00 * oldDPtr[bigOffset + backSheetOffset] +
					w01 * oldDPtr[bigOffset + ind_p01] +
					w10 * oldDPtr[bigOffset + ind_p10] +
					w11 * oldDPtr[bigOffset + ind_p11];
			}
			else{
				newDPtr[numSheet * newSheetCount + offset] = 0;
			}
		}
	}

	template <typename Dtype>
	void InterpImageNN_gpu(const Blob<Dtype>* orig, const float* coord,
		Blob<Dtype>* warped, const Interp& interp){
		//Get the paramters from the original and warped and apply the
		//transformation to it.
		const Dtype* orgDataPtr = orig->gpu_data();
		Dtype* warpedDataPtr = warped->mutable_gpu_data();
		int oldNPerSheet = orig->height() * orig->width();
		int newNPerSheet = warped->height() * warped->width();
		int nCount = warped->count();
		switch (interp)
		{
		case NN:
			nn_interpolation_kernel<Dtype> << <CAFFE_GET_BLOCKS(nCount),
				CAFFE_CUDA_NUM_THREADS >> >(nCount, orgDataPtr, oldNPerSheet, 
				warpedDataPtr, newNPerSheet, coord);
			break;
		case BILINEAR:
			bilinear_interpolation_kernel<Dtype> << <CAFFE_GET_BLOCKS(nCount),
				CAFFE_CUDA_NUM_THREADS >> >(nCount, orgDataPtr, oldNPerSheet, warpedDataPtr, 
				newNPerSheet, coord, orig->width());
			break;
		default:
			LOG(ERROR) << "Unkown interpolation mode " << interp;
			break;
		}
		CUDA_POST_KERNEL_CHECK;
	}

	//explicit instantiation
	template void InterpImageNN_gpu(const Blob<float>* orig, const float* coord,
		Blob<float>* warped, const Interp& interp);
//	template void InterpImageNN_gpu(const Blob<double>* orig, const float* coord,
//		Blob<double>* warped, const Interp& interp);

	/*
	 *******PropagateErrorNN_gpu********
	 *If we kernalize each pixel in the top(warped image), bc of race conditions
	 *we need to use atomaticAdd, but it's slow and there is no double implementation
	 *of atomicAdd.
	 *So instead, parallelize over each pixel in the bottom (original) and for each pixel
	 * loop over the coord, find those top neurons that came from this bottom pixel and add.
	 * Similar to MaxPoolBackward Super. fucking slow. duh.
	 */
	template <typename Dtype>
	__global__ void PropagateErrorNN_kernel_nonatomic(
		const int nthreads, const Dtype* top_diff, const int width,
		const int height, const int channels, const int num,
		const int top_len, const float* coord, Dtype* bottom_diff){
		int index = blockIdx.x * blockDim.x + threadIdx.x;
		if (index < nthreads){
			//find out the target index to look for in coord
			//can do this the way abhishek did so we can save on some computation(like
			//with SheetCount)
			int w = index % width;
			int h = (index / width) % height;
			int c = (index / width / height) % channels;
			int n = index / width / height / channels;

			int target_ind = h * width + w;
			//move over top_diff ptr to the beginning of its hxw sheet:
			//top_len = width_top * height_top
			top_diff += (n * channels + c) * top_len;

			Dtype gradient = 0;
			//loop over coord and add to grad if coord[i] == target_ind
			for (int i = 0; i < top_len; ++i){
				gradient += top_diff[i] * (static_cast<int>(coord[i]) == target_ind);
			}
			bottom_diff[index] += gradient;
		}
	}

	template <typename Dtype>
	__global__ void nn_backpropagation_kernel(int nThreads, const Dtype* topDataPtr,
		int topSheetCount, Dtype* bottomDataPtr,
		int bottomSheetCount, const float* coord){
		int index = blockIdx.x * blockDim.x + threadIdx.x;
		if (index < nThreads){
			int offset = index % topSheetCount;
			int numSheet = index / topSheetCount;

			int bottomSheetOffset = static_cast<int>(coord[offset]);
			if (bottomSheetOffset >= 0){
				int bottomFinalOffset = numSheet * bottomSheetCount + bottomSheetOffset;
				//AJ: as atomicAdd is only available to float, this only works if
				//Dtype = float
				atomicAdd((&bottomDataPtr[bottomFinalOffset]),
					static_cast<float>(topDataPtr[numSheet * topSheetCount + offset]));
			}
		}
	}

	template <typename Dtype>
	__global__ void bilinear_backpropagation_kernel(int nThreads, const Dtype* topDataPtr,
		int topSheetCount, Dtype* bottomDataPtr, int bottomSheetCount,
		const float* coord, int W){
		int index = blockIdx.x * blockDim.x + threadIdx.x;
		if (index < nThreads){
			int offset = index % topSheetCount;
			int numSheet = index / topSheetCount;
			int bottomSheetOffset = static_cast<int>(coord[offset]);
			if (bottomSheetOffset >= 0){
				int c0 = bottomSheetOffset % W;
				int ind_p11 = static_cast<int>(coord[offset + topSheetCount]);
				int c1 = ind_p11 % W;

				int ind_p01 = bottomSheetOffset - c0 + c1; //r0 * W + c1
				int ind_p10 = ind_p11 - c1 + c0;

				float dc = coord[offset + 2 * topSheetCount];
				float dr = coord[offset + 3 * topSheetCount];

				float w00 = (1 - dc)*(1 - dr);
				float w01 = (1 - dr)*dc;
				float w10 = (1 - dc)*dr;
				float w11 = dr * dc;

				float top_error = static_cast<float>(topDataPtr[index]);

				int commonOffset = numSheet * bottomSheetCount;

				//p00
				atomicAdd((&bottomDataPtr[commonOffset + bottomSheetOffset]),
					w00 * top_error);
				//p01
				atomicAdd((&bottomDataPtr[commonOffset + ind_p01]), w01 * top_error);
				//p10
				atomicAdd((&bottomDataPtr[commonOffset + ind_p10]), w10 * top_error);
				//p11
				atomicAdd(&bottomDataPtr[commonOffset + ind_p11], w11 * top_error);
			}
		}
	}

	template <typename Dtype>
	void BackPropagateErrorNN_gpu(const Blob<Dtype>* top, const float* coord,
		Blob<Dtype>* bottom, const Interp &interp){
	    //Get the parameters from the original and warped and apply the 
		//transformation to it.
		const Dtype* topDataPtr = top->gpu_diff();
		Dtype* bottomDataPtr = bottom->mutable_gpu_diff();
		int topNPerSheet = top->height() * top->width();
		int bottomNPerSheet = bottom->height() * bottom->width();
		//atomicAdd needs nTop many threads
		int nCount = top->count();
		switch (interp)
		{
		case NN:
			nn_backpropagation_kernel<Dtype> << <CAFFE_GET_BLOCKS(nCount),
				CAFFE_CUDA_NUM_THREADS >> >(nCount, topDataPtr,
				topNPerSheet, bottomDataPtr, bottomNPerSheet, coord);
			break;
		case BILINEAR:
			bilinear_backpropagation_kernel<Dtype> << <CAFFE_GET_BLOCKS(nCount),
				CAFFE_CUDA_NUM_THREADS >> >(nCount, topDataPtr,
				topNPerSheet, bottomDataPtr, bottomNPerSheet, coord, bottom->width());
			break;
		default:
			LOG(ERROR) << "Unknown interpolation mode " << interp;
			break;
		}
		CUDA_POST_KERNEL_CHECK;
	}

	//explicit instantiation
	template void BackPropagateErrorNN_gpu(const Blob<float>* top, const float* coord,
		Blob<float>* bottom, const Interp &interp);
//	template void PropagateErrorNN_gpu(const Blob<double>* top, const float* coord,
//		Blob<double>* bottom, const Interp &interp);

}