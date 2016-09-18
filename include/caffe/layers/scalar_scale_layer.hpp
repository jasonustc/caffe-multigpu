#ifndef CAFFE_SCALAR_SCALE_LAYER_HPP_
#define CAFFE_SCALAR_SCALE_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/layers/neuron_layer.hpp"

namespace caffe {

/**
 * @brief scale the input by a fixed scalar, this is some thing like scale_layer
 *        and bias_layer
 *        but the scale, bias are manually set and fixed
 *       output := scale_ * input + bias_
 */
template <typename Dtype>
class ScalarScaleLayer : public NeuronLayer<Dtype> {
public:
	explicit ScalarScaleLayer(const LayerParameter& param)
		: NeuronLayer<Dtype>(param) {}
	virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "ScalarScale"; }
  // Scale
  virtual inline int ExactBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  Dtype scale_;
  Dtype bias_;
};
}  // namespace caffe

#endif  // CAFFE_SCALAR_SCALE_LAYER_HPP_
