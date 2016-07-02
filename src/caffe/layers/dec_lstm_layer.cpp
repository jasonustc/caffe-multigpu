#include<string>
#include<utility>
#include<vector>

#include "caffe/layers/dec_lstm_layer.hpp"

namespace caffe{
	/*
	 * bottom[0]->shape(1): #streams
	 */
	template <typename Dtype>
	void DLSTMLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top){
	}
}