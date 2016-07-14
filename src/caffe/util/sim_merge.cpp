#include <utility> 
#include <vector>
#include "caffe/util/sim_merge.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe{
	template <typename Dtype>
	void update_sim_matrix_cpu(Blob<Dtype>* weight, Blob<Dtype>* sim,
		const int axis){
		//dim_0 * dim_1 * ... * dim_{axis_-1} is the number of output
		const int N = weight->count(0, axis);
		//dim_{axis} * dim_{axis+1} * ... is the number of weights for a single output
		const int K = weight->count(axis);
		CHECK_GT(K, 1) << "similarity can only be computed between vectors";
		const vector<int> sim_shape(2, N);
		sim->Reshape(sim_shape);
		const Dtype* weight_data = weight->mutable_cpu_data();
		Dtype* sim_data = sim->mutable_cpu_data();
		caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, N, N, K, Dtype(1),
			weight_data, weight_data, Dtype(0), sim_data);
		//sim(\vec{a}, \vec{b}) = (\vec{a} \dot \vec{b}) / 
		//(\sqrt(\vec{a} \dot \vec{a}) \times \sqrt(\vec{b} \dot \vec{b})
		for (int i = 0; i < N; ++i){
			for (int j = i + 1; j < N; ++j){
				const Dtype sqrt_i = sqrt(sim_data[i * N + i]);
				const Dtype sqrt_j = sqrt(sim_data[j * N + j]);
				const Dtype denom = sqrt_i * sqrt_j;
				sim_data[i * N + j] /= denom;
				sim_data[j * N + i] /= denom;
			}
		}
		// set similarity to itself 0
		for (int i = 0; i < N; ++i){
			sim_data[i * N + i] = 0;
		}
	}

	template void update_sim_matrix_cpu<float>(Blob<float>* weight, Blob<float>* sim,
		const int axis);
	template void update_sim_matrix_cpu<double>(Blob<double>* weight, Blob<double>* sim,
		const int axis);

	template <typename Dtype>
	void refresh_weight_cpu(const int j, Blob<Dtype>* weight,
		Filler<Dtype>* filler, const int dim){
		Dtype* weight_data = weight->mutable_cpu_data();
		Dtype* weight_data_j = weight_data + j * dim;
		// re-initialize by filler
		filler->Fill(dim, weight_data_j);
	}

	template void refresh_weight_cpu<double>(int j, Blob<double>* weight,
		Filler<double>* filler, const int dim);
	template void refresh_weight_cpu<float>(int j, Blob<float>* weight,
		Filler<float>* filler, const int dim);

	template <typename Dtype>
	void merge_sim_weights_cpu(Blob<Dtype>* weight, Blob<Dtype>* sim,
		const Dtype prop, Filler<Dtype>* filler, const int axis, string name){
		// get similarity matrix
		//dim_0 * dim_1 * ... * dim_{axis_-1} is the number of output
		const int N = weight->count(0, axis);
		//dim_{axis} * dim_{axis+1} * ... is the number of weights for a single output
		const int K = weight->count(axis);
		CHECK_GT(K, 1) << "similarity can only be computed between vectors";
		update_sim_matrix_cpu(weight, sim, axis);
		Dtype* weight_data = weight->mutable_cpu_data();
		Dtype* sim_data = sim->mutable_cpu_data();
		Dtype* sim_temp = sim->mutable_cpu_diff();
		// here the value in sim_data will be changed in nth_element,
		// so we need to save sim data in diff for temporary usage
		caffe_copy<Dtype>(weight->count(), sim_data, sim_temp);
		//get similarity threshold
		const int nth = prop * N * N;
		std::nth_element(sim_temp, sim_temp + nth, sim_temp + N * N, std::greater<Dtype>());
		const Dtype sim_th = sim_temp[nth];
		std::set<int> merged_index;
		for (int i = 0; i < N; ++i){
			if (merged_index.count(i)){
				continue;
			}
			for (int j = i + 1; j < N; ++j){
				if (merged_index.count(j)){
					continue;
				}
				const Dtype sim_ij = sim_data[i * N + j];
				if (sim_ij > sim_th){
					// NOTE: other options: 
					//   1. merge the pair with the largest similairty
					//   2. merge muliple pairs in a time 
					//weight_i := (1 - sim_ij) * weight_i + sim_ij * weight_j
					caffe_cpu_axpby<Dtype>(K, Dtype(sim_ij), weight_data + j * K,
						Dtype(1 - sim_ij), weight_data + i * K);
					// NOTE: diff will be cleared in solver for all learnable params_
					// so it's not necessary to merge the difference here
					//refresh weight
					refresh_weight_cpu(j, weight, filler, K);
					merged_index.insert(i);
					merged_index.insert(j);
					break;
				}
			}//for (int j = i + 1; j < N; ++j)
		}//for (int i = 0; i < N; ++i)
		const Dtype prop_merged = Dtype(merged_index.size()) / Dtype(N) / Dtype(2);
		LOG(INFO) << prop_merged << " of the weights in \"" << name << "\" are merged";
	}

	template void merge_sim_weights_cpu<float>(Blob<float>* weight,
		Blob<float>* sim, const float prop, 
		Filler<float>* filler, const int axis, string name);
	template void merge_sim_weights_cpu<double>(Blob<double>* weight, 
		Blob<double>* sim, const double prop, 
		Filler<double>* filler, const int axis, string name);
}// namespace caffe
