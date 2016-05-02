#ifndef CAFFE_PATCH_RANK_LAYER_HPP_
#define CAFFE_PATCH_RANK_LAYER_HPP_

#include "caffe/common.hpp"
#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe{
	/**
	 * @brief rerank feature map patches, to achieve global invariance
	 **/
	template <typename Dtype>
	class PatchRankLayer : public Layer<Dtype>{
	public:
		explicit PatchRankLayer(const LayerParameter& param)
			: Layer<Dtype>(param){}
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual inline const char* type() const { return "ReOrder"; }
		virtual inline int ExactNumBottomBlobs() const { return 1; }
		virtual inline int ExactNumTopBlobs() const { return 1; }

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

		/**
		 * @brief get energy of given block by norm of activations
		 **/
		virtual void GetBlockEnergy_cpu(const vector<Blob<Dtype>*>& bottom);
		//not support virtual function?
//		virtual void GetBlockEnergy_gpu(const vector<Blob<Dtype>*>& bottom);
		void GetBlockEnergy_gpu(const vector<Blob<Dtype>*>& bottom);

		/**
		 * @brief first split image[start_x, start_y, end_x, end_y] 
		 * into blok_num_ blocks
		 * then reorder these blocks based on their energies
		 **/
		virtual void GetBlockOffset_cpu();
		void GetBlockOffset_gpu();


		int pyramid_height_;
		int split_num_;
		int unit_block_width_;
		int unit_block_height_;
		int num_unit_block_;
		int num_;
		int channels_;
		PatchRankParameter_EnergyType energy_type_;


		/*
		 * shape: num_ * channels_ * num_unit_block_ * num_unit_block_
		 * offset w after patch rank will be saved into data
		 * offset h after patch rank will be saved into diff
		 */
		Blob<Dtype> block_offsets_;

		/*
		 * energies of each unit block 
		 * shape: num_ * channels_ * num_unit_block_ * num_unit_block_
		 */
		Blob<Dtype> block_energies_;
	};
}


#endif // CAFFE_PATCH_RANK_LAYER_HPP_