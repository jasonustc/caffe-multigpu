#include <vector>
#include "caffe/layers/gan_gate_layer.hpp"

namespace caffe{
	template <typename Dtype>
	void GANGateLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top){
		// gan_mode_ in [1, 2, 3]
		gan_mode_ = 1;
		top[0]->ReshapeLike(*bottom[0]);
	}

	template <typename Dtype>
	void GANGateLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top){
		CHECK_EQ(bottom[0]->num(), bottom[1]->num());
		CHECK_EQ(bottom[0]->channels(), bottom[1]->channels());
		CHECK_EQ(bottom[0]->height(), bottom[1]->height());
		CHECK_EQ(bottom[0]->width(), bottom[1]->width());
		// 1: true 2,3: fake
		int index = gan_mode_ == 1 ? 0 : 1;
		top[0]->ReshapeLike(*bottom[index]);
		top[0]->ShareData(*bottom[index]);
		top[0]->ShareDiff(*bottom[index]);
    LOG(INFO) << "bottom index: " << index;
    LOG(INFO) << "top data in " << this->layer_param_.name();
    for (int i = 0; i < 10; ++i){
      std::cout << top[0]->cpu_data()[i] << " ";
    }
    std::cout << std::endl;
    for (int i = 0; i < 10; ++i){
      std::cout << top[0]->cpu_data()[2*3*128*128 + i] << " ";
    }
    std::cout << std::endl;
    // update gan_mode_
    gan_mode_ = gan_mode_ == 3 ? 1 : gan_mode_ + 1;
	}

	INSTANTIATE_CLASS(GANGateLayer);
	REGISTER_LAYER_CLASS(GANGate);
} // namespace caffe
