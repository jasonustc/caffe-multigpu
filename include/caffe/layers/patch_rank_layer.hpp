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

		virtual inline const char* type() const { return "PatchRank"; }
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
		//energy of each block is summed across feature maps
		void MergeEnergyAcrossMaps_cpu();
		//not support virtual function?
//		virtual void GetBlockEnergy_gpu(const vector<Blob<Dtype>*>& bottom);
		void GetBlockEnergy_gpu(const vector<Blob<Dtype>*>& bottom);
		void MergeEnergyAcrossMaps_gpu();

		/**
		 * @brief first split image[start_x, start_y, end_x, end_y] 
		 * into blok_num_ blocks
		 * then reorder these blocks based on their energies
		 **/
		void GetBlockOffset_cpu();
		void SortBlock_gpu();
		void ComputeLevelOffset_gpu();
		void MergeOffset_gpu();


		int pyramid_height_;
		int split_num_;
		int unit_block_width_;
		int unit_block_height_;
		int num_unit_block_;
		int num_;
		int channels_;
		//if the order across feature maps need to be consistent
		bool consistent_;
		PatchRankParameter_EnergyType energy_type_;

		/*
		 * To store energies for GPU computation
		 * data: sum of energy of each block
		 * diff: indexes
		 */
		vector<Blob<Dtype>*> block_infos_;

		/*
		 * shape: num_ * channels_ * 
		 *            * [(split_num * split_num)^pyramid_height_ 
	     *            - (split_num * split_num)]  
	     *            / (split_num * split_num - 1)
		 * offset w after patch rank will be saved into data
		 * offset h after patch rank will be saved into diff
		 */
		//NOTE: maybe int is more efficient for memory usage
		vector<Blob<Dtype>*> block_offsets_;

		Blob<Dtype> test_data_;

	};
}


#endif // CAFFE_PATCH_RANK_LAYER_HPP_