/********************************************************************************
** Copyright(c) 2016 USTC All Rights Reserved.
** auth£º Xu Shen
** mail£º shenxuustc@gmail.com
** date£º 2016/07/02
** desc£º SeqEndLayer
*********************************************************************************/
#include <utility>
#include <vector>

#include "caffe/layer.hpp"
#include "boost/scoped_ptr.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/blob.hpp"

namespace caffe{
	/**
	 * @brief get the end of input sequence
	 * NOTE: two assumptions:
	 *       1. the first element is the beginning of a sequence
	 *       2. the last element is the end of a sequence
	 * used for decoding LSTM or prediction LSTM
	 **/
	template <typename Dtype>
	class SeqEndLayer : public Layer<Dtype>{
	public:
		explicit SeqEndLayer(const LayerParameter& param)
			: Layer<Dtype>(param){}
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual inline int ExactNumBottomBlobs() const { return 2; }
		virtual inline int ExactNumTopBlobs() const { return 1; }
		virtual inline bool AllowForceBackward(const int bottom_index) const{
			// Can't propagate to sequence continuation indicators
			return bottom_index != 1;
		}
		inline void InferSeqEndId(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);

	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down,
			const vector<Blob<Dtype>*>& bottom);
		virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down,
			const vector<Blob<Dtype>*>& bottom);

		//number of sequences
		vector<int> end_id_;
	};

}// namespace caffe