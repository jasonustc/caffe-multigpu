#include <string>
#include <utility>
#include <vector>

#include "caffe/layers/dec_lstm_unit_layer.hpp"

namespace caffe{
	template <typename Dtype>
	__device__ Dtype sigmoid(Dtype x){
		return Dtype(1) / (Dtype(1) + exp(-x));
	}

	template <typename Dtype>
	__device__ Dtype tanh(Dtype x){
		return Dtype(2) * sigmoid(Dtype(2) * x) - Dtype(1);
	}

	template <typename Dtype>
	__device__ Dtype relu(Dtype x){
		return x > 0 ? x : 0;
	}

	template <typename Dtype>
	__global__ void DLSTMActsForward(const int nthreads, const int dim,
		const Dtype* X, Dtype* X_acts){
		CUDA_KERNEL_LOOP(index, nthreads){
			const int x_dim = 4 * dim;
			const int d = index % x_dim;
			if (d < 3 * dim){
				X_acts[index] = sigmoid(X[index]);
			}
			else{
				X_acts[index] = tanh(X[index]);
			}
		}
	}

	template <typename Dtype>
	__global__ void DLSTMUnitForward(const int nthreads, const int dim,
		const Dtype* C_prev, const Dtype* X, Dtype* C, Dtype* H){
		CUDA_KERNEL_LOOP(index, nthreads){
			const int n = index / dim;
			const int d = index % dim;
			const Dtype* X_offset = X + 4 * dim * n;
			const Dtype i = X_offset[d];
			const Dtype f = X_offset[dim + d];
			const Dtype o = X_offset[2 * dim + d];
			const Dtype g = X_offset[3 * dim + d];
			const Dtype c = f * C_prev[d] + i * g;
			C[d] = c;
			const Dtype tanh_c = tanh(c);
			H[d] = o * tanh_c;
		}
	}

	template <typename Dtype>
	void DLSTMUnitLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top){
		const Dtype* C_prev = bottom[0]->gpu_data();
		const Dtype* X = bottom[1]->gpu_data();
		Dtype* X_acts = X_acts_.mutable_gpu_data();
		Dtype* C = top[0]->mutable_gpu_data();
		Dtype* H = top[1]->mutable_gpu_data();
		const int count = bottom[0]->count();
		const int x_count = bottom[1]->count();
		DLSTMActsForward<Dtype> << <CAFFE_GET_BLOCKS(x_count), CAFFE_CUDA_NUM_THREADS >> >(
			x_count, hidden_dim_, X, X_acts);
		CUDA_POST_KERNEL_CHECK;
		DLSTMUnitForward<Dtype> << < CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS >> >(
			count, hidden_dim_, C_prev, X_acts, C, H);
		CUDA_POST_KERNEL_CHECK;
	}

	template <typename Dtype>
	__global__ void DLSTMUnitBackward(const int nthreads, const int dim,
		const Dtype* C_prev, const Dtype* X, const Dtype* C,
		const Dtype* C_diff, const Dtype* H_diff, Dtype* C_prev_diff, Dtype* X_diff){
		CUDA_KERNEL_LOOP(index, nthreads){
			const int n = index / dim;
			const int d = index % dim;
			const Dtype* X_offset = X + 4 * dim * n;
			const Dtype i = X_offset[d];
			const Dtype f = X_offset[dim + d];
			const Dtype o = X_offset[2 * dim + d];
			const Dtype g = X_offset[3 * dim + d];
			const Dtype c = C[d];
			const Dtype tanh_c = tanh(c);
			Dtype* X_diff_offset = X_diff + 4 * dim * n;
			Dtype* i_diff = X_diff_offset + d;
			Dtype* f_diff = X_diff_offset + dim + d;
			Dtype* o_diff = X_diff_offset + 2 * dim + d;
			Dtype* g_diff = X_diff_offset + 3 * dim + d;
			Dtype* c_prev_diff = C_prev_diff + d;
			Dtype h_diff = H_diff[d];
			Dtype c_diff = C_diff[d];
			//accumulate diff bp from c_t and h_t
			const Dtype c_term_diff = c_diff + h_diff * (1 - tanh_c * tanh_c);
			*c_prev_diff = c_term_diff * f;
			*g_diff = c_term_diff * i;
			*o_diff = tanh_c * h_diff;
			*f_diff = c_term_diff * C_prev[d];
			*i_diff = c_term_diff * g;
		}
	}

	template <typename Dtype>
	__global__ void DLSTMActsBackward(const int nthreads, const int dim,
		const Dtype* X_acts, const Dtype* X_acts_diff, Dtype* X_diff){
		CUDA_KERNEL_LOOP(index, nthreads){
			const int x_dim = 4 * dim;
			const int d = index % x_dim;
			const Dtype X_act = X_acts[index];
			if (d < 3 * dim){
				X_diff[index] = X_acts_diff[index] * X_act * (Dtype(1) - X_act);
			}
			else{
				X_diff[index] = X_acts_diff[index] * (Dtype(1) - X_act * X_act);
			}
		}
	}

	template <typename Dtype>
	void DLSTMUnitLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom){
		const Dtype* C_prev = bottom[0]->gpu_data();
		const Dtype* X = bottom[1]->gpu_data();
		Dtype* X_acts = X_acts_.mutable_gpu_data();
		const Dtype* H_diff = top[1]->gpu_diff();
		const Dtype* C_diff = top[0]->gpu_diff();
		const Dtype* C = top[0]->gpu_data();
		Dtype* X_diff = bottom[1]->mutable_gpu_diff();
		Dtype* X_acts_diff = X_acts_.mutable_gpu_diff();
		Dtype* C_prev_diff = bottom[0]->mutable_gpu_diff();
		const int count = top[1]->count();
		const int x_count = bottom[1]->count();
		//why do forward of action of X again?
		DLSTMActsForward<Dtype> << <CAFFE_GET_BLOCKS(x_count), CAFFE_CUDA_NUM_THREADS >> >(
			x_count, hidden_dim_, X, X_acts);
		CUDA_POST_KERNEL_CHECK;
		DLSTMUnitBackward<Dtype> // NOLINT_NEXT_LINE(whitespace/operators)
			<< <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>> > (
			count, hidden_dim_, C_prev, X, C, C_diff, H_diff, C_prev_diff, X_acts_diff);
		CUDA_POST_KERNEL_CHECK;
		DLSTMActsBackward<Dtype> // NOLINT_NEXT_LINE(whitespace/operators)
			<< <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>> > (
			x_count, hidden_dim_, X_acts, X_acts_diff, X_diff);
		CUDA_POST_KERNEL_CHECK;
	}

	INSTANTIATE_LAYER_GPU_FUNCS(DLSTMUnitLayer);

} // namespace caffe
