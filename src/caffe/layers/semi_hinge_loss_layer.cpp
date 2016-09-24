#include <vector>
#include <utility>
#include <math.h>

#include "caffe/layers/semi_hinge_loss_layer.hpp"

namespace caffe {
	template <typename Dtype>
	void SemiHingeLossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top){
		LossLayer<Dtype>::LayerSetUp(bottom, top);
		ignore_label_ = this->layer_param_.semi_hinge_loss_param().ignore_label();
		sup_bias_ = this->layer_param_.semi_hinge_loss_param().sup_bias();
		unsup_bias_ = this->layer_param_.semi_hinge_loss_param().unsup_bias();
		gamma_ = this->layer_param_.semi_hinge_loss_param().gamma();
		sup_thre_ = this->layer_param_.semi_hinge_loss_param().sup_thre();
		unsup_thre_ = this->layer_param_.semi_hinge_loss_param().unsup_thre();
		axis_ = this->layer_param_.semi_hinge_loss_param().axis();
		axis_ = bottom[0]->CanonicalAxisIndex(axis_);
		vector<int> bottom_shape = bottom[0]->shape();
		for (size_t i = 0; i < bottom_shape.size(); ++i){
			CHECK_EQ(bottom_shape[i], bottom[1]->shape(i));
		}
		CHECK_EQ(bottom[2]->count(), bottom[3]->count());
		CHECK_EQ(bottom[2]->count(), bottom[0]->count(0, axis_));
		diff_.reset(new Blob<Dtype>(bottom[0]->shape()));
		vector<int> dist_shape(1, bottom[0]->count(0, axis_));
		dist_.reset(new Blob<Dtype>(dist_shape));
	}

	template <typename Dtype>
	void SemiHingeLossLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top){
		LossLayer<Dtype>::Reshape(bottom, top);
		vector<int> bottom_shape = bottom[0]->shape();
		for (size_t i = 0; i < bottom_shape.size(); ++i){
			CHECK_EQ(bottom_shape[i], bottom[1]->shape(i));
		}
		CHECK_EQ(bottom[2]->count(), bottom[3]->count());
		CHECK_EQ(bottom[2]->count(), bottom[0]->count(0, axis_));
		diff_->ReshapeLike(*bottom[0]);
		vector<int> dist_shape(1, bottom[0]->count(0, axis_));
		dist_->Reshape(dist_shape);
	}

	template <typename Dtype>
	void SemiHingeLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top){
		const int count = bottom[0]->count();
		const int dim = bottom[0]->count(axis_);
		const int num = count / dim;
		const Dtype* x1_data = bottom[0]->cpu_data();
		const Dtype* x2_data = bottom[1]->cpu_data();
		Dtype* diff_data = diff_->mutable_cpu_data();
		const Dtype* label_data_1 = bottom[2]->cpu_data();
		const Dtype* label_data_2 = bottom[3]->cpu_data();
		// loss in dist_data
		Dtype* dist_data = dist_->mutable_cpu_data();
		// buff for dist
		Dtype* dist_diff = dist_->mutable_cpu_diff();
		Dtype loss(0.);
		caffe_sub(count, x1_data, x2_data, diff_data);
		for (int n = 0; n < num; ++n){
			const int label_1 = static_cast<int>(label_data_1[n]);
			const int label_2 = static_cast<int>(label_data_2[n]);
			// D(x_1, x_2)
			const Dtype dist = caffe_cpu_dot(dim, diff_data, diff_data);
			if (label_1 != ignore_label_ && label_2 != ignore_label_){
				// indicator
				const int ind = label_1 == label_2 ? 1 : -1;
				// supervised hinge loss
				dist_data[n] = std::max(Dtype(0), sup_bias_ - ind * (sup_thre_ - dist));
				loss += dist_data[n];
			}
			else{
				// unsupervised hinge loss
				dist_data[n] = gamma_ * std::max(Dtype(0), unsup_bias_ - abs(unsup_thre_ - dist));
				// will be used in backward propagation
				dist_diff[n] = dist;
				loss += dist_data[n];
			}
			diff_data += dim;
		}
		top[0]->mutable_cpu_data()[0] = loss / num / Dtype(2);
	}

	template <typename Dtype>
	void SemiHingeLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down,
		const vector<Blob<Dtype>*>& bottom){
		Dtype* bottom_diff_1 = bottom[0]->mutable_cpu_diff();
		Dtype* bottom_diff_2 = bottom[1]->mutable_cpu_diff();
		const Dtype* label_data_1 = bottom[2]->cpu_data();
		const Dtype* label_data_2 = bottom[3]->cpu_data();
		const Dtype* diff_data = diff_->cpu_data();
		const Dtype* dist_data = dist_->cpu_data();
		const Dtype* dist_diff = dist_->cpu_diff();
		const int num = bottom[0]->count(0, axis_);
		const int dim = bottom[0]->count(axis_);
		Dtype loss_weight = top[0]->cpu_diff()[0];
		for (int n = 0; n < num; ++n){
			const int label_1 = static_cast<int>(label_data_1[n]);
			const int label_2 = static_cast<int>(label_data_2[n]);
			if (label_1 != ignore_label_ && label_2 != ignore_label_){
				// indicator
				const int ind = label_1 == label_2 ? 1 : -1;
				// supervised gradient
				if (dist_data[n] > 0){
					if (propagate_down[0]){
						caffe_cpu_axpby(dim, Dtype(ind * loss_weight / num), diff_data,
							Dtype(0), bottom_diff_1);
					}
					if (propagate_down[1]){
						caffe_cpu_axpby(dim, Dtype(-ind * loss_weight / num), diff_data,
							Dtype(0), bottom_diff_2);
					}
				}
			}
			else{
				// unsupervised gradient
				if (dist_data[n] > 0){
					Dtype gap = dist_diff[n] - unsup_thre_;
					Dtype alpha = gap > 0 ? 1 : (gap == 0 ? 0 : -1);
					if (propagate_down[0]){
						caffe_cpu_axpby(dim, Dtype(gamma_ * alpha * loss_weight / num), diff_data,
							Dtype(0), bottom_diff_1);
					}
					if (propagate_down[1]){
						caffe_cpu_axpby(dim, Dtype(-gamma_ * alpha * loss_weight / num), diff_data,
							Dtype(0), bottom_diff_2);
					}
				}
			}
			diff_data += dim;
			bottom_diff_1 += dim;
			bottom_diff_2 += dim;
		}
	}

#ifdef CPU_ONLY
STUB_GPU(SemiHingeLossLayer);
#endif

	INSTANTIATE_CLASS(SemiHingeLossLayer);
	REGISTER_LAYER_CLASS(SemiHingeLoss);
} // namespace caffe