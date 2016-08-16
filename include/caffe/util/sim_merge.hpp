#ifndef CAFFE_UTIL_SIM_MERGE_HPP_
#define CAFFE_UTIL_SIM_MERGE_HPP_
#include <vector>
#include <utility>

#include "caffe/blob.hpp"
#include "caffe/filler.hpp"
#include "caffe/common.hpp"

namespace caffe{

	template <typename Dtype>
	void update_sim_matrix_cpu(Blob<Dtype>* weight, Blob<Dtype>* sim,
		const int axis);

	template <typename Dtype>
	void refresh_weight_cpu(const int j, Blob<Dtype>* weight,
		Filler<Dtype>* filler, const int dim);

	template <typename Dtype>
	void merge_sim_weights_cpu(Blob<Dtype>* weight, Blob<Dtype>* sim,
		const Dtype prop, Filler<Dtype>* filler, const int axis, string name,
		const bool hard = false);

	template <typename Dtype>
	void update_sim_matrix_gpu(Blob<Dtype>* weight, Blob<Dtype>* sim,
		const int axis);

	template <typename Dtype>
	void merge_sim_weights_gpu(Blob<Dtype>* weight, Blob<Dtype>* sim,
		const Dtype prop, Filler<Dtype>* filler, const int axis, string name, 
		const bool hard = false);

} // namespace caffe

#endif// CAFFE_SIM_MERGE_HPP_