#include "caffe/layers/axis_pooling_layer.hpp"

namespace caffe {

	template <typename Dtype>
	void AxisPoolingLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		Dtype* top_data = top[0]->mutable_gpu_data();
		const Dtype* bottom_data = bottom[0]->gpu_data();
		const int bottom_pool_axis = bottom[0]->shape(pool_axis_);

		int* mask = NULL;

		switch (this->layer_param_.axis_pooling_param().pool())
		{
		case AxisPoolingParameter_PoolMethod_MAX:
			mask = max_idx_.mutable_gpu_data();
			caffe_gpu_set(max_idx_.count(), -1, mask);
			caffe_gpu_set(top[0]->count(), Dtype(-FLT_MAX), top_data);
			for (int n = 0; n < num_pools_; n++)
			{
				for (int i = 0; i < bottom_pool_axis; i++)
				{
					for (int j = 0; j < pool_input_size_; j++)
					{
						const int bottom_index = n*bottom_pool_axis*pool_input_size_ + i*pool_input_size_ + j;
						const int top_index = n*pool_input_size_ + j;
						if (bottom_data[bottom_index]>top_data[top_index])
						{
							top_data[top_index] = bottom_data[bottom_index];
							mask[top_index] = bottom_index;
						}
					}
				}
			}
			break;
		case AxisPoolingParameter_PoolMethod_AVE:
			caffe_gpu_set(top[0]->count(), Dtype(0), top_data);
			for (int n = 0; n < num_pools_; ++n) {
				for (int i = 0; i < bottom_pool_axis; i++)
				{
					caffe_gpu_add(pool_input_size_,
						bottom_data + n * bottom_pool_axis * pool_input_size_ + i*pool_input_size_,
						top_data + n*pool_input_size_,
						top_data + n*pool_input_size_
						);
				}
			}
			caffe_gpu_scal(top[0]->count(), Dtype(1) / bottom_pool_axis, top_data);
			break;
		case AxisPoolingParameter_PoolMethod_STOCHASTIC:
			NOT_IMPLEMENTED;
		default:
			LOG(FATAL) << "Unknown pooling method.";
		}
	}

	template <typename Dtype>
	void AxisPoolingLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
		if (!propagate_down[0])
			return;

		const Dtype* top_diff = top[0]->gpu_diff();
		Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
		const int bottom_pool_axis = bottom[0]->shape(pool_axis_);
		caffe_gpu_set(bottom[0]->count(), Dtype(0), bottom_diff);
		const int* mask = NULL;

		switch (this->layer_param_.axis_pooling_param().pool())
		{
		case AxisPoolingParameter_PoolMethod_MAX:
			mask = max_idx_.gpu_data();
			for (int n = 0; n < num_pools_; n++)
			{
				for (int j = 0; j < pool_input_size_; j++)
				{
					const int top_index = n*pool_input_size_ + j;
					const int bottom_index = mask[top_index];
					bottom_diff[bottom_index] = top_diff[top_index];
				}
			}
			break;
		case AxisPoolingParameter_PoolMethod_AVE:
			for (int n = 0; n < num_pools_; ++n) {
				for (int i = 0; i < bottom_pool_axis; i++)
				{
					caffe_copy(pool_input_size_,
						top_diff + n*pool_input_size_,
						bottom_diff + n * bottom_pool_axis * pool_input_size_ + i*pool_input_size_
						);
				}
			}
			break;
		case AxisPoolingParameter_PoolMethod_STOCHASTIC:
			NOT_IMPLEMENTED;
		default:
			LOG(FATAL) << "Unknown pooling method.";
		}
	}

	INSTANTIATE_LAYER_GPU_FUNCS(AxisPoolingLayer);
}  // namespace caffe