#include "caffe/layers/axis_pooling_layer.hpp"

namespace caffe {

	template <typename Dtype>
	void AxisPoolingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
	}

	template <typename Dtype>
	void AxisPoolingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {

		const AxisPoolingParameter& AxisPooling_param = this->layer_param_.axis_pooling_param();

		pool_axis_ = bottom[0]->CanonicalAxisIndex(AxisPooling_param.axis());
		num_pools_ = bottom[0]->count(0, pool_axis_);
		pool_input_size_ = bottom[0]->count(pool_axis_ + 1);

		vector<int> shape;
		shape.push_back(num_pools_);
		shape.push_back(pool_input_size_);
		top[0]->Reshape(shape);
		// If max pooling, we will initialize the vector index part.
		if (this->layer_param_.pooling_param().pool() ==
			PoolingParameter_PoolMethod_MAX) {
			max_idx_.Reshape(shape);
		}
	}

	template <typename Dtype>
	void AxisPoolingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		Dtype* top_data = top[0]->mutable_cpu_data();
		const Dtype* bottom_data = bottom[0]->cpu_data();
		const int bottom_pool_axis = bottom[0]->shape(pool_axis_);

		int* mask = NULL;

		//loop at the pool axis
		switch (this->layer_param_.axis_pooling_param().pool())
		{
		case AxisPoolingParameter_PoolMethod_MAX:
			mask = max_idx_.mutable_cpu_data();
			caffe_set(max_idx_.count(), -1, mask);
			caffe_set(top[0]->count(), Dtype(-FLT_MAX), top_data);
			for (int n = 0; n < num_pools_; n++)
			{
				for (int i = 0; i < bottom_pool_axis; i++)
				{
					for (int j = 0; j < pool_input_size_; j++)
					{
						const int bottom_index = n * bottom_pool_axis * pool_input_size_ + i * pool_input_size_ + j;
						const int top_index = n * pool_input_size_ + j;
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
			caffe_set(top[0]->count(), Dtype(0), top_data);
			for (int n = 0; n < num_pools_; ++n) {
				for (int i = 0; i < bottom_pool_axis; i++)
				{
					caffe_add(pool_input_size_,
						bottom_data + n * bottom_pool_axis * pool_input_size_ + i * pool_input_size_,
						top_data + n * pool_input_size_,
						top_data + n * pool_input_size_);
				}
			}
			caffe_scal(top[0]->count(), Dtype(1) / bottom_pool_axis, top_data);
			break;
		case AxisPoolingParameter_PoolMethod_STOCHASTIC:
			NOT_IMPLEMENTED;
		default:
			LOG(FATAL) << "Unknown pooling method.";
		}
	}

	template <typename Dtype>
	void AxisPoolingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
		if (!propagate_down[0])
			return;

		const Dtype* top_diff = top[0]->cpu_diff();
		Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
		const int bottom_pool_axis = bottom[0]->shape(pool_axis_);
		caffe_set(bottom[0]->count(), Dtype(0), bottom_diff);
		const int* mask = NULL;

		switch (this->layer_param_.axis_pooling_param().pool())
		{
		case AxisPoolingParameter_PoolMethod_MAX:
			mask = max_idx_.cpu_data();
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

#ifdef CPU_ONLY
	STUB_GPU(AxisPoolingLayer);
#endif

	INSTANTIATE_CLASS(AxisPoolingLayer);
	REGISTER_LAYER_CLASS(AxisPooling);

}  // namespace caffe