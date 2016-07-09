#include <string>
#include <utility>
#include <vector>

#include "caffe/layers/dec_lstm_unit_layer.hpp"

namespace caffe{
	template <typename Dtype>
	inline Dtype sigmoid(Dtype x){
		return 1. / (1. + exp(-x));
	}

	template <typename Dtype>
	inline Dtype tanh(Dtype x){
		return 2. * sigmoid(2. * x) - 1.;
	}

	template <typename Dtype>
	inline Dtype relu(Dtype x){
		return (x > 0 ? x : 0);
	}

	template <typename Dtype>
	void DLSTMUnitLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top){
		const int num_instances = bottom[0]->shape(1);
		for (int i = 0; i < bottom.size(); ++i){
			// 1 x N x D/4D
			CHECK_EQ(3, bottom[i]->num_axes());
			CHECK_EQ(1, bottom[i]->shape(0));
			CHECK_EQ(num_instances, bottom[i]->shape(1));
		}
		hidden_dim_ = bottom[0]->shape(2);
		CHECK_EQ(4 * hidden_dim_, bottom[1]->shape(2));
		top[0]->ReshapeLike(*bottom[0]);
		top[1]->ReshapeLike(*bottom[0]);
		X_acts_.ReshapeLike(*bottom[1]);
	}

	template <typename Dtype>
	void DLSTMUnitLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top){
		//#streams
		const int num = bottom[0]->shape(1);
		const int x_dim = 4 * hidden_dim_;
		const Dtype* C_prev = bottom[0]->cpu_data();
		const Dtype* X = bottom[1]->cpu_data();
		Dtype* C = top[0]->mutable_cpu_data();
		Dtype* H = top[1]->mutable_cpu_data();
		for (int n = 0; n < num; ++n){
			for (int d = 0; d < hidden_dim_; ++d){
				const Dtype i = sigmoid(X[d]);
				const Dtype f = sigmoid(X[hidden_dim_ + d]);
				const Dtype o = sigmoid(X[2 * hidden_dim_ + d]);
				const Dtype g = tanh(X[3 * hidden_dim_ + d]);
				const Dtype c = f * C_prev[d] + i * g;
				C[d] = c;
				const Dtype tanh_c = tanh(c);
				H[d] = o * tanh_c;
			}
			C += hidden_dim_;
			C_prev += hidden_dim_;
			H += hidden_dim_;
			X += x_dim;
		}
	}

	template <typename Dtype>
	void DLSTMUnitLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom){
		//#streams
		const int num = bottom[0]->shape(1);
		const int x_dim = 4 * hidden_dim_;
		const Dtype* C_prev = bottom[0]->cpu_data();
		const Dtype* X = bottom[1]->cpu_data();
		Dtype* C = top[0]->mutable_cpu_data();
		Dtype* H = top[1]->mutable_cpu_data();
		const Dtype* C_diff = top[0]->cpu_diff();
		const Dtype* H_diff = top[1]->cpu_diff();
		Dtype* C_prev_diff = bottom[0]->mutable_cpu_diff();
		Dtype* X_diff = bottom[1]->mutable_cpu_diff();
		for (int n = 0; n < num; ++n){
			for (int d = 0; d < hidden_dim_; ++d){
				const Dtype i = sigmoid(X[d]);
				const Dtype f = sigmoid(X[hidden_dim_ + d]);
				const Dtype o = sigmoid(X[2 * hidden_dim_ + d]);
				const Dtype g = tanh(X[3 * hidden_dim_ + d]);
				const Dtype c = C[d];
				const Dtype tanh_c = tanh(c);
				Dtype* i_diff = X_diff + d;
				Dtype* f_diff = X_diff + hidden_dim_ + d;
				Dtype* o_diff = X_diff + 2 * hidden_dim_ + d;
				Dtype* g_diff = X_diff + 3 * hidden_dim_ + d;
				Dtype* c_prev_diff = C_prev_diff + d;
				Dtype h_diff = H_diff[d];
				Dtype c_diff = C_diff[d];
				//accumulate diff bp from c_t and h_t
				const Dtype c_term_diff = c_diff + h_diff * (1 - tanh_c * tanh_c);
				*c_prev_diff = c_term_diff * f;
				*g_diff = c_term_diff * i * (1 - g * g);
				*o_diff = (1 - o) * o * tanh_c * h_diff;
				*f_diff = c_term_diff * C_prev[d] * f * (1 - f);
				*i_diff = c_term_diff * g * i * (1 - i);
			}
			C += hidden_dim_;
			C_prev += hidden_dim_;
			H += hidden_dim_;
			X += x_dim;
			H_diff += hidden_dim_;
			C_diff += hidden_dim_;
			X_diff += x_dim;
			C_prev_diff += hidden_dim_;
		}
	}

	INSTANTIATE_CLASS(DLSTMUnitLayer);
} // namespace caffe
