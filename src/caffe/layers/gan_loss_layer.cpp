#include <algorithm>
#include <vector>

#include "caffe/layers/gan_loss_layer.hpp"


namespace caffe{
	template <typename Dtype>
	void GANLossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top){
		LossLayer<Dtype>::LayerSetUp(bottom, top);
		iter_idx_ = 0;
		dis_iter_ = this->layer_param_.gan_loss_param().dis_iter();
		gen_iter_ = this->layer_param_.gan_loss_param().gen_iter();

		// discriminative mode
		if (bottom.size() == 2){
			CHECK_EQ(bottom[0]->shape(0), bottom[1]->shape(0));
			CHECK_EQ(bottom[0]->shape(1), 1);
			CHECK_EQ(bottom[1]->shape(1), 1);
		}
		// generative mode
		if (bottom.size() == 1){
			CHECK_EQ(bottom[0]->shape(1), 1);
		}
	}

	template <typename Dtype>
	void GANLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top){
		int batch_size = bottom[0]->count();
		Dtype loss(0.);
		//1. discriminative mode
		if (bottom.size() == 2){
			const Dtype* score1 = bottom[0]->cpu_data();
			const Dtype* score2 = bottom[1]->cpu_data();
			for (int i = 0; i < batch_size; ++i){
				// generally sigmoid, so no 0s or 1s
				loss -= log(score1[i]) + log(1 - score2[i]);
			}
		}
		//2. generative mode
		if (bottom.size() == 1){
			const Dtype* score = bottom[0]->cpu_data();
			for (int i = 0; i < batch_size; ++i){
				loss -= log(score[i]);
			}
		}
		loss /= batch_size;
		top[0]->mutable_cpu_data()[0] = loss;
		iter_idx_++;
	}

	template <typename Dtype>
	void GANLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom){
		int batch_size = bottom[0]->count();
		//1. discriminative mode
		if (bottom.size() == 2){
			if (iter_idx_ % dis_iter_ == 0){
				for (int i = 0; i < batch_size; ++i){
					bottom[0]->mutable_cpu_diff()[i] = Dtype(-1.) /
						bottom[0]->cpu_data()[i] / Dtype(batch_size);
					bottom[1]->mutable_cpu_diff()[i] = Dtype(-1.) /
						(bottom[1]->cpu_data()[i] - Dtype(1.)) / Dtype(batch_size);
				}
			}
			else{
				caffe_set<Dtype>(bottom[0]->count(), Dtype(0.), bottom[0]->mutable_cpu_diff());
				caffe_set<Dtype>(bottom[1]->count(), Dtype(0.), bottom[1]->mutable_cpu_diff());
			}
		}
		// 2. generative mode
		if (bottom.size() == 1){
			if (iter_idx_ % gen_iter_ == 0){
				for (int i = 0; i < batch_size; ++i){
					bottom[0]->mutable_cpu_diff()[i] = Dtype(-1.) /
						bottom[0]->cpu_data()[i] / Dtype(batch_size);
				}
			}
			else{
				caffe_set<Dtype>(bottom[0]->count(), Dtype(0.), bottom[0]->mutable_cpu_diff());
			}
		}
	}

	INSTANTIATE_CLASS(GANLossLayer);
	REGISTER_LAYER_CLASS(GANLoss);

	template <typename Dtype>
	void GANDGLossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top){
		LossLayer<Dtype>::LayerSetUp(bottom, top);
		diter_idx_ = 0;
		giter_idx_ = 0;
		dis_iter_ = this->layer_param_.gan_loss_param().dis_iter();
		gen_iter_ = this->layer_param_.gan_loss_param().gen_iter();
		gan_mode_ = 1;
	}

	template <typename Dtype>
	void GANDGLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top){
		int batch_size = bottom[0]->count();
		const Dtype* score = bottom[0]->cpu_data();
		Dtype loss(0.);
		// when gan_mode_ == 1, this input of loss is D(x)
		// loss is discriminative loss: -log(D(x))
		if (gan_mode_ == 1){
			diter_idx_++;
			for (int i = 0; i < batch_size; ++i){
				loss -= log(score[i]);
			}
		}
		// when gan_mode_ == 2, the input of the loss is D(G(z))
		// loss is discriminative loss: -log(1-D(G(z)))
		else if (gan_mode_ == 2){
			for (int i = 0; i < batch_size; ++i){
				loss -= log(1 - score[i]);
			}
		}
		// when gan_mode_ == 3, the input of the loss is D(G(z))
		// loss is generative loss: -log(D(G(z)))
		else if (gan_mode_ == 3){
			giter_idx_++;
			for (int i = 0; i < batch_size; ++i){
				loss -= log(score[i]);
			}
		}
		loss /= Dtype(batch_size);
		top[0]->mutable_cpu_data()[0] = loss;
	}

	template <typename Dtype>
	void GANDGLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom){
		int batch_size = bottom[0]->count();
		// when gan_mode_ == 1, this input of loss is D(x)
		// backward for discriminative loss
		if (gan_mode_ == 1){
			if (diter_idx_ % dis_iter_ == 0){
				for (int i = 0; i < batch_size; ++i){
					bottom[0]->mutable_cpu_data()[i] = Dtype(-1) /
						bottom[0]->cpu_data()[i] / Dtype(batch_size);
				}
			}
			else{
				caffe_set<Dtype>(bottom[0]->count(), Dtype(0.), bottom[0]->mutable_cpu_diff());
			}
		}
		// when gan_mode_ == 2, this input of loss is D(G(z))
		// backward for discriminative loss
		else if (gan_mode_ == 2){
			if (diter_idx_ % dis_iter_ == 0){
				for (int i = 0; i < batch_size; ++i){
					bottom[0]->mutable_cpu_diff()[i] = Dtype(-1) /
						(bottom[0]->cpu_data()[i] - Dtype(1)) / Dtype(batch_size);
				}
			}
			else{
				caffe_set<Dtype>(bottom[0]->count(), Dtype(0.), bottom[0]->mutable_cpu_diff());
			}
		}
		// when gan_mode_ == 3, this input of loss is D(G(z))
		// backward for generative loss
		else if (gan_mode_ == 3){
			if (giter_idx_ % gen_iter_ == 0){
				for (int i = 0; i < batch_size; ++i){
					bottom[0]->mutable_cpu_diff()[i] = Dtype(-1) /
						bottom[0]->cpu_data()[i] / Dtype(batch_size);
				}
			}
			else{
				caffe_set<Dtype>(bottom[0]->count(), Dtype(0.), bottom[0]->mutable_cpu_diff());
			}
		}
	}

	INSTANTIATE_CLASS(GANDGLossLayer);
	REGISTER_LAYER_CLASS(GANDGLoss);

	template <typename Dtype>
	void WGANLossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top){
		LossLayer<Dtype>::LayerSetUp(bottom, top);
		diter_idx_ = 0;
		giter_idx_ = 0;
		dis_iter_ = this->layer_param_.gan_loss_param().dis_iter();
		gen_iter_ = this->layer_param_.gan_loss_param().gen_iter();
		gan_mode_ = 1;
		CHECK_EQ(bottom[0]->shape(1), 1);
	}

	template <typename Dtype>
	void WGANLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top){
		int batch_size = bottom[0]->count();
		const Dtype* score = bottom[0]->cpu_data();
		Dtype loss(0.);
		// when gan_mode_ == 1, this input of loss is D(x)
		// loss is discriminative loss: D(x)
		if (gan_mode_ == 1){
			diter_idx_++;
			for (int i = 0; i < batch_size; ++i){
				loss += score[i];
			}
		}
		// when gan_mode_ == 2, the input of the loss is D(G(z))
		// loss is discriminative loss: -D(G(z))
		else if (gan_mode_ == 2){
			for (int i = 0; i < batch_size; ++i){
				loss -= score[i];
			}
		}
		// when gan_mode_ == 3, the input of the loss is D(G(z))
		// loss is generative loss: D(G(z))
		else if (gan_mode_ == 3){
			giter_idx_++;
			for (int i = 0; i < batch_size; ++i){
				loss += score[i];
			}
		}
		loss /= Dtype(batch_size);
		top[0]->mutable_cpu_data()[0] = loss;
	}

	template <typename Dtype>
	void WGANLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom){
		int batch_size = bottom[0]->count();
		// when gan_mode_ == 1, this input of loss is D(x)
		// backward for discriminative loss
		if (gan_mode_ == 1){
			if (diter_idx_ % dis_iter_ == 0){
				for (int i = 0; i < batch_size; ++i){
					bottom[0]->mutable_cpu_diff()[i] = Dtype(1) / Dtype(batch_size);
				}
			}
			else{
				caffe_set<Dtype>(bottom[0]->count(), Dtype(0.), bottom[0]->mutable_cpu_diff());
			}
		}
		// when gan_mode_ == 2, this input of loss is D(G(z))
		// backward for discriminative loss
		else if (gan_mode_ == 2){
			if (diter_idx_ % dis_iter_ == 0){
				for (int i = 0; i < batch_size; ++i){
					bottom[0]->mutable_cpu_diff()[i] = Dtype(-1) / Dtype(batch_size);
				}
			}
			else{
				caffe_set<Dtype>(bottom[0]->count(), Dtype(0.), bottom[0]->mutable_cpu_diff());
			}
		}
		// when gan_mode_ == 3, this input of loss is D(G(z))
		// backward for generative loss
		else if (gan_mode_ == 3){
			if (giter_idx_ % gen_iter_ == 0){
				for (int i = 0; i < batch_size; ++i){
					bottom[0]->mutable_cpu_diff()[i] = Dtype(1) / Dtype(batch_size);
				}
			}
			else{
				caffe_set<Dtype>(bottom[0]->count(), Dtype(0.), bottom[0]->mutable_cpu_diff());
			}
		}
		// update gan_mode_
		gan_mode_ = gan_mode_ == 3 ? 1 : gan_mode_ + 1;
	}

	INSTANTIATE_CLASS(WGANLossLayer);
	REGISTER_LAYER_CLASS(WGANLoss);
}
