#include <vector>
#include <utility>

#include "caffe/layers/local_lstm_layer.hpp"

namespace caffe{
	template <typename Dtype>
	void LocalLSTMLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top){
		LSTMLayer<Dtype>::LayerSetUp(bottom, top);
		vector<int> x_shape(3, 1);
		x_shape[1] = bottom[0]->shape(1);
		x_shape[2] = bottom[0]->shape(2);

		// predict of x before and after activation
		xp_.reset(new Blob<Dtype>(x_shape));
		xp_act_.reset(new Blob<Dtype>(x_shape));

		Blob<Dtype> test_blob;
		test_blob.ToTxt("my_test", true);

		// setup predict innerproduct layer
		LayerParameter ip_xp_param;
		// filler setting
		ip_xp_param.CopyFrom(this->layer_param_.inner_product_param());
		// axis and num_output
		ip_xp_param.mutable_inner_product_param()->set_axis(2);
		ip_xp_param.mutable_inner_product_param()->set_num_output(x_shape[2]);
		ip_xp_.reset(new InnerProductLayer<Dtype>(ip_xp_param));
		const vector<Blob<Dtype>*> ip_xp_bottom(1, H_[0].get());
		const vector<Blob<Dtype>*> ip_xp_top(1, xp_.get());
		ip_xp_->SetUp(ip_xp_bottom, ip_xp_top);

		// setup activation layer
		switch (this->layer_param_.recurrent_param().act_type()){
		case RecurrentParameter_ActType_RELU:
			act_.reset(new ReLULayer<Dtype>(LayerParameter()));
			break;
		case RecurrentParameter_ActType_SIGMOID:
			act_.reset(new SigmoidLayer<Dtype>(LayerParameter()));
			break;
		default:
			LOG(FATAL) << "Unkown activation type";
		}
		// NOTE: maybe a in-place operation is enough
		const vector<Blob<Dtype>*> act_bottom(1, xp_.get());
		const vector<Blob<Dtype>*> act_top(1, xp_act_.get());
		act_->SetUp(act_bottom, act_top);
		
		// local learning parameters
	}
} // namespace caffe