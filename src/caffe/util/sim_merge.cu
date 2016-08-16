#include "caffe/util/sim_merge.hpp"
#include <thrust/sort.h>
#include <thrust/functional.h>

namespace caffe{

	template <typename Dtype>
	__global__ void ComputeSim(const int count, const int N, Dtype* sim_data){
		CUDA_KERNEL_LOOP(index, count){
			const int row = index / N;
			const int col = index % N;
			//sim(\vec{a}, \vec{b}) = (\vec{a} \dot \vec{b}) / 
			//(\sqrt(\vec{a} \dot \vec{a}) \times \sqrt(\vec{b} \dot \vec{b})
			const Dtype sqrt_i = sqrt(sim_data[row * N + row]);
			const Dtype sqrt_j = sqrt(sim_data[col * N + col]);
			const Dtype denom = sqrt_i * sqrt_j;
			sim_data[row * N + col] /= denom;
		}
	}

	// set diagonal elements to 0.
	template <typename Dtype>
	__global__ void ResetDiag(const int N, Dtype* sim_data){
		CUDA_KERNEL_LOOP(index, N){
			sim_data[index * N + index] = Dtype(0);
		}
	}

	template <typename Dtype>
	void update_sim_matrix_gpu(Blob<Dtype>* weight, 
		Blob<Dtype>* sim, const int axis){
		//dim_0 * dim_1 * ... * dim_{axis_-1} is the number of output
		const int N = weight->count(0, axis);
		//dim_{axis} * dim_{axis+1} * ... is the number of weights for a single output
		const int K = weight->count(axis);
		CHECK_GT(K, 1) << "similarity can only be computed between vectors";
		// N x N
		const vector<int> sim_shape(2, N);
		sim->Reshape(sim_shape);
		Dtype* weight_data = weight->mutable_gpu_data();
		//to save memory, put history similarity in data
		//and current similarity in diff
		Dtype* sim_data = sim->mutable_gpu_data();
		caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, N, N, K, Dtype(1.),
			weight_data, weight_data, Dtype(0), sim_data);
		const int count = N * N;
		ComputeSim<Dtype> // NOLINT_NEXT_LINE(whitespace/operators)
			<< < CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS >> >
			(count, N, sim_data);
		CUDA_POST_KERNEL_CHECK;
		ResetDiag<Dtype> // NOLINT_NEXT_LINE(whitespace/operators)
			<< < CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS >> >
			(N, sim_data);
		CUDA_POST_KERNEL_CHECK;
	}

	template void update_sim_matrix_gpu<double>(Blob<double>* weight, 
		Blob<double>* sim, const int axis);
	template void update_sim_matrix_gpu<float>(Blob<float>* weight, 
		Blob<float>* sim, const int axis);

	template <typename Dtype>
	void merge_sim_weights_gpu(Blob<Dtype>* weight, Blob<Dtype>* sim,
		const Dtype prop, Filler<Dtype>* filler, const int axis, string name,
		const bool hard){
		// get similarity matrix
		//dim_0 * dim_1 * ... * dim_{axis_-1} is the number of output
		const int N = weight->count(0, axis);
		//dim_{axis} * dim_{axis+1} * ... is the number of weights for a single output
		const int K = weight->count(axis);
		CHECK_GT(K, 1) << "similarity can only be computed between vectors";
		update_sim_matrix_gpu(weight, sim, axis);
		Dtype* weight_data = weight->mutable_gpu_data();
		Dtype* sim_data = sim->mutable_gpu_data();
		Dtype* sim_temp = sim->mutable_gpu_diff();
		Dtype sim_th;
		//get similarity threshold
		const Dtype* sim_data_cpu = sim->cpu_data();
		if (hard){
			// hard: use fixed threshold
			sim_th = prop;
		}
		else{
			// soft: use adaptive threshold
			const int nth = prop * N * N;
			// here the value in sim_data will be changed, so we need to save history
			// similarities in diff for backup
			caffe_copy<Dtype>(sim->count(), sim_data, sim_temp);
			// not working?
			//thrust::sort(sim_temp, sim_temp + sim->count(), thrust::greater<Dtype>());
			Dtype* sim_temp_cpu = sim->mutable_cpu_diff();
			std::nth_element(sim_temp_cpu, sim_temp_cpu + nth,
				sim_temp_cpu + N * N, std::greater<Dtype>());
			sim_th = sim_temp_cpu[nth];
		}
		std::set<int> merged_pos_index;
		LOG(INFO) << "merge positive correlated weights:";
		for (int i = 0; i < N; ++i){
			if (merged_pos_index.count(i)){
				continue;
			}
			for (int j = i + 1; j < N; ++j){
				if (merged_pos_index.count(j)){
					continue;
				}
				const Dtype sim_ij = sim_data_cpu[i * N + j];
				if (sim_ij > sim_th){
					// NOTE: other options: 
					//   1. merge the pair with the largest similairty
					//   2. merge muliple pairs in a time 
					//weight_i := (1 - sim_ij) * weight_i + sim_ij * weight_j
					caffe_gpu_axpby<Dtype>(K, Dtype(sim_ij), weight_data + j * K,
						Dtype(1 - sim_ij), weight_data + i * K);
					// NOTE: diff will be cleared in solver for all learnable params_
					// so it's not necessary to merge the difference here
					//refresh weight
					refresh_weight_cpu(j, weight, filler, K);
					merged_pos_index.insert(i);
					merged_pos_index.insert(j);
					LOG(INFO) << "weight_" << i << " and weight_" << j;
					break;
				}
			}//for (int j = i + 1; j < N; ++j)
		}//for (int i = 0; i < N; ++i)
		Dtype prop_merged = Dtype(merged_pos_index.size()) / Dtype(N) / Dtype(2);
		LOG(INFO) << prop_merged << " of the weights in \"" << name << "\" are merged(pos)";
		std::set<int> merged_neg_index;
		LOG(INFO) << "enhance negitive correlated weights:";
		for (int i = 0; i < N; ++i){
			if (merged_neg_index.count(i)){
				continue;
			}
			for (int j = i + 1; j < N; ++j){
				if (merged_neg_index.count(j)){
					continue;
				}
				const Dtype sim_ij = sim_data_cpu[i * N + j];
				if (sim_ij < -sim_th){
					// NOTE: other options: 
					//   1. merge the pair with the lowest similairty
					//   2. merge muliple pairs in a time 
					// weight_i := (1 - sim_ij) * weight_i + sim_ij * weight_j
					caffe_gpu_axpby<Dtype>(K, Dtype(sim_ij), weight_data + j * K,
						Dtype(1 - sim_ij), weight_data + i * K);
					// NOTE: diff will be cleared in solver for all learnable params_
					// so it's not necessary to merge the difference here
					// negative correlation we substract the correlated and enhance
					// its own weights instead of just randomly initialize one of them
					caffe_gpu_axpby<Dtype>(K, Dtype(sim_ij), weight_data + i * K,
						Dtype(1 - sim_ij), weight_data + j * K);
					// refresh weight
					// refresh_weight_cpu(j, weight, filler, K);
					merged_neg_index.insert(i);
					merged_neg_index.insert(j);
					LOG(INFO) << "weight_" << i << " and weight_" << j;
					break;
				}
			}//for (int j = i + 1; j < N; ++j)
		}//for (int i = 0; i < N; ++i)
		prop_merged = Dtype(merged_neg_index.size()) / Dtype(N) / Dtype(2);
		LOG(INFO) << prop_merged << " of the weights in \"" << name << "\" are enhanced(neg)";
	}

	template void merge_sim_weights_gpu<float>(Blob<float>* weight, Blob<float>* sim,
		const float prop, Filler<float>* filler, const int axis, string name,
		const bool hard);
	template void merge_sim_weights_gpu<double>(Blob<double>* weight, Blob<double>* sim,
		const double prop, Filler<double>* filler, const int axis, string name,
		const bool hard);
}// namespace caffe