/********************************************************************************
** Copyright(c) 2015 USTC Reserved.
** auth: Xu Shen
** mail: shenxu@mail.ustc.edu.cn
** date: 2015/11/4
** desc: Image Transformation Functions(GPU && CPU)
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
#ifndef CAFFE_UTIL_TRANSFORM_H_
#define CAFFE_UTIL_TRANSFORM_H_

#include <limits>
#include "caffe/blob.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe{
	const float PI_F = 3.14159265358979f;

	enum Direction {RIGHT, LEFT};
	enum TransType {ROTATION, SCALE, SHIFT};

	//Crop is zero-padding, CLAMP is border replicate, REFLECT is mirror.
	enum Border {CROP, CLAMP, REFLECT};
	enum Interp {NN, BILINEAR};

	//compute parameters in transformation matrix by given transform type
	void TMatFromParam(TransType transType, const float param1, const float param2, float *tmat, bool invert = false);

	//matrix is multiplied to the existing one from the right
	void AddRotation(const float &angle, float *mat, const Direction dir = RIGHT);
	void AddScale(const float &scale, float* mat, const Direction dir = RIGHT);
	void AddShift(const float &dx, const float& dy, float *mat, const Direction dir = RIGHT);

	//m = m * t
	void AddTransform(float *mat, const float *tmp, const Direction dir = RIGHT);

	void Invert3x3(float *A);

	void generate_nn_coord(const int &height, const int &width,
		const int &height_new,const int &width_new, const Border &border, 
		const float* coord_data_res, float* &coord_data);

	void generate_bilinear_coord(const int &height, const int &width,
		const int &height_new, const int &width_new,
		const Border &border, const float *coord_data_res,
		float *&coord_data);

	//This one doesn't change the size.
	void GenCoordMatCrop_cpu(Blob<float>& tmat,
		const int &height, const int &width, Blob<float>& ori_coord,
		Blob<float>& coord_idx, const Border &border = CROP, const Interp &interp = NN);

	//This one doesn't change the size.
	void GenCoordMatCrop_gpu(Blob<float>& tmat, const int height, const int width,
		Blob<float>& ori_coord, Blob<float>& coord_idx, const Border& border, const Interp& interp);

	//Generates identity coordinates (y, x, 1).
	void GenBasicCoordMat(float *coord, const int &width, const int &height);

	template <typename Dtype> void Reflect(Dtype &val, const int size);
	template <typename Dtype> void Clamp(Dtype &val, const int size);

	template <typename Dtype>
	void InterpImageNN_cpu(const Blob<Dtype> *orig, const float *coord,
		Blob<Dtype> *warped, const Interp &interp = NN);

	template <typename Dtype>
	void nn_interpolation(const Blob<Dtype> *&orig, const float *&coord,
		Blob<Dtype> *&warped);

	template <typename Dtype>
	void bilinear_interpolation(const Blob<Dtype>* orig, const float* coord,
		Blob<Dtype>* warped);

	template <typename Dtype>
	void BackPropagateErrorNN_cpu(const Blob<Dtype> *top, const float *coord,
		Blob<Dtype> *bottom, const Interp &interp = NN);

	template <typename Dtype>
	void nn_backpropagation(const Blob<Dtype> *&top, const float *&coord,
		Blob<Dtype>* &bottom);

	template <typename Dtype>
	void bilinear_backpropagation(const Blob<Dtype>* & top, const float* & coord,
		Blob<Dtype>* &bottom);

	//gpu functions
	template <typename Dtype>
	void InterpImageNN_gpu(const Blob<Dtype> *orig, const float *coord,
		Blob<Dtype> *warped, const Interp &interp = NN);

	template <typename Dtype>
	void BackPropagateErrorNN_gpu(const Blob<Dtype> *top, const float *coord,
		Blob<Dtype> *bottom, const Interp &interp = NN);

}//namespace caffe

#endif