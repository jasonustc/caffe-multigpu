/********************************************************************************
** Copyright(c) 2016 USTC Reserved.
** auth: Xu Shen
** mail: shenxu@mail.ustc.edu.cn
** date: 2016/04/29
** desc: PatchRankLayer(GPU)
*********************************************************************************/
#include "caffe/layers/patch_rank_layer.hpp"
#include <thrust/sort.h>

namespace caffe{

	/*
	 * nthreads: total number of unit blocks
	 * num_ * channels_ * num_unit_block_ * num_unit_block_
	 */
	template<typename Dtype>
	__global__ void ComputeBlockEnergyL1(const int nthreads, int split_num,
		const int height, const int width, const int unit_block_height,
		const int unit_block_width, const int num_unit_block, const Dtype* bottom_data,
		Dtype* energy_data, Dtype* index_data){
		CUDA_KERNEL_LOOP(index, nthreads){
			int bw = index % num_unit_block;
			int bh = (index / num_unit_block) % num_unit_block;
			int c = index /  num_unit_block / num_unit_block;
			Dtype sum = 0;
			for (int h = 0; h < unit_block_height; ++h){
				for (int w = 0; w < unit_block_width; ++w){
					int offset = c * height * width + bh * unit_block_height * width 
						+ bw * unit_block_width;
					sum += abs(bottom_data[offset + h * width + w]);
				}
			}
			// save data of a single block for next level(split_num * split_num)
			// in continuous memory, which makes sorting faster
			int bw_next = bw / split_num;
			int bh_next = bh / split_num;
			int block_size = split_num * split_num;
			int next_block_offset = bh_next * split_num * block_size + 
				bw_next * block_size;
			int next_inner_bw = bw % split_num;
			int next_inner_bh = bh % split_num;
			energy_data[c * num_unit_block * num_unit_block + 
				next_block_offset + next_inner_bh * split_num + 
				next_inner_bw] = sum;
			index_data[c * num_unit_block * num_unit_block + 
				next_block_offset + next_inner_bh * split_num + 
				next_inner_bw] = bh * num_unit_block + bw;
		}
	}

	/*
	 * nthreads: total number of unit blocks
	 * num_ * channels_ * num_unit_block_ * num_unit_block_
	 */
	template<typename Dtype>
	__global__ void ComputeBlockEnergyL2(const int nthreads, int split_num,
		const int height, const int width, const int unit_block_height,
		const int unit_block_width, const int num_unit_block, const Dtype* bottom_data,
		Dtype* energy_data, Dtype* index_data){
		CUDA_KERNEL_LOOP(index, nthreads){
			int bw = index % num_unit_block;
			int bh = (index / num_unit_block) % num_unit_block;
			int c = index / num_unit_block/ num_unit_block;
			Dtype sum = 0;
			for (int h = 0; h < unit_block_height; ++h){
				for (int w = 0; w < unit_block_width; ++w){
					int offset = ( c * height + bh * unit_block_height)
						* width + bw * unit_block_width;
					sum += bottom_data[offset + h * width + w] * 
						bottom_data[offset + h * width + w];
				}
			}
			// save data of a single block for next level(split_num * split_num)
			// in continuous memory, which makes sorting faster
			int bw_next = bw / split_num;
			int bh_next = bh / split_num;
			int block_size = split_num * split_num;
			int next_block_offset = bh_next * split_num * block_size + 
				bw_next * block_size;
			int next_inner_bw = bw % split_num;
			int next_inner_bh = bh % split_num;
			energy_data[c * num_unit_block * num_unit_block + 
				next_block_offset + next_inner_bh * num_unit_block + 
				next_inner_bw] = sum;
			index_data[c * num_unit_block * num_unit_block + 
				next_block_offset + next_inner_bh * num_unit_block +  
				next_inner_bw] = bh * num_unit_block + bw;
		}
	}

  /*
   * nthreads = num_ * channels_ * block_num * block_num
   * each thread get energy of blockes of given pyramid level
   * one single block in this level corresponds to split_num * split_num
   * inner blocks in the previous level
   */
	template<typename Dtype>
	__global__ void ComputeLevelEnergy(int nthreads, int split_num,
		int block_num, const Dtype* prev_level_energy,
		Dtype* level_energy, Dtype* level_index){
		CUDA_KERNEL_LOOP(index, nthreads){
			int patch_num = block_num * split_num;
			int bw = index % block_num;
			int bh = (index / block_num) % block_num;
			int c = index / block_num / block_num;
			//offset in energy map of previous level
			int prev_block_size = split_num * split_num;
			int patch_offset = c * patch_num * patch_num + bh * split_num *
				prev_block_size + bw * prev_block_size;
			Dtype sum = 0;
			//sum in "patch" of this level, computed from previous level data
			for (int h = 0; h < split_num; ++h){
				for (int w = 0; w < split_num; ++w){
					sum += prev_level_energy[patch_offset + h * split_num + w];
				}
			}
			// save data of a single block for next level(split_num * split_num)
			// in continuous memory, which makes sorting faster
			int bw_next = bw / split_num;
			int bh_next = bh / split_num;
			int block_size = split_num * split_num;
			int next_block_offset = bh_next * split_num * block_size + 
				bw_next * block_size;
			int next_inner_bw = bw % split_num;
			int next_inner_bh = bh % split_num;
			level_energy[c * block_num * block_num + 
				next_block_offset + next_inner_bh * block_num + 
				next_inner_bw] = sum;
			level_index[c * block_num * block_num + 
				next_block_offset + next_inner_bh * split_num + 
				next_inner_bw] = bh * block_num + bw;
		}
	}

	template<typename Dtype>
	void PatchRankLayer<Dtype>::GetBlockEnergy_gpu(const vector<Blob<Dtype>*>& bottom){
		const Dtype* bottom_data = bottom[0]->gpu_data();
		Dtype* energy_data = block_infos_[0]->mutable_gpu_data();
		Dtype* index_data = block_infos_[0]->mutable_gpu_diff();
		const int width = bottom[0]->width();
		const int height = bottom[0]->height();
		int count = num_ * channels_ * num_unit_block_ * num_unit_block_;
		switch (energy_type_){
		case PatchRankParameter_EnergyType_L1:
			ComputeBlockEnergyL1<Dtype> << <CAFFE_GET_BLOCKS(count), 
				CAFFE_CUDA_NUM_THREADS >> >(count, split_num_,
				height, width, unit_block_height_, unit_block_width_,
				num_unit_block_, bottom_data, energy_data, index_data);
			CUDA_POST_KERNEL_CHECK;
			break;
		case PatchRankParameter_EnergyType_L2:
			ComputeBlockEnergyL2<Dtype> << <CAFFE_GET_BLOCKS(count),
				CAFFE_CUDA_NUM_THREADS >> >(count, split_num_, 
				height, width, unit_block_height_, unit_block_width_,
				num_unit_block_, bottom_data, energy_data, index_data);
			CUDA_POST_KERNEL_CHECK;
			break;
		default:
			LOG(FATAL) << "Unkown energy type.";
		}
		cudaStream_t* stream = new cudaStream_t[pyramid_height_ - 1];
		//lunch multi-kernel
		for (int i = 0; i < pyramid_height_ - 1; ++i){
			cudaStreamCreate(&stream[i]);
		}
//		block_infos_[0]->ToTxt("block_info_0", true);
		for (int p = 1; p < pyramid_height_; ++p){
			int count = block_infos_[p]->count();
			int block_num = block_infos_[p]->width();
			const Dtype* prev_level_energy = block_infos_[p - 1]->gpu_data();
			Dtype* level_energy = block_infos_[p]->mutable_gpu_data();
			Dtype* level_index = block_infos_[p]->mutable_gpu_diff();
			ComputeLevelEnergy<Dtype> << <1, CAFFE_GET_BLOCKS(count),
				CAFFE_CUDA_NUM_THREADS, stream[p - 1] >> >(count, split_num_,
				block_num, prev_level_energy, level_energy, level_index);
			CUDA_POST_KERNEL_CHECK;
//			ostringstream oss;
//			oss << p;
//			block_infos_[p]->ToTxt("block_info_" + oss.str(), true);
		}
		for (int i = 0; i < pyramid_height_ - 1; ++i){
			cudaStreamDestroy(stream[i]);
		}
	}

	template<typename Dtype>
	__device__ void swap(Dtype* data, const int i, const int j){
		Dtype tmp = data[i];
		data[i] = data[j];
		data[j] = tmp;
	}

	/*
	* because the vec will be quite small (split_num_ * split_num_)
	* and generally split_num_ will be set to 2 or 3 or 4
	* we can use bubble sort algorithm
	* both sort on values and indexes
	* in descend order
	*/
	template<typename Dtype>
	__device__ void bubble_sort(const int n, Dtype* values, Dtype* indexes){
		for (int i = 0; i < n; ++i){
			bool swapped = false;
			for (int j = 0; j < n - (i + 1); ++j){
				if (values[j] < values[j + 1]){
					swap<Dtype>(values, j, j + 1);
					swap<Dtype>(indexes, j, j + 1);
					swapped = true;
				}
			}
			if (!swapped){ break; }
		}
	}

	/*
	 * nthreads = num_ * channels_ * next_block_num * next_block_num
	 * the rank in level p will determine the offset of level p - 1
	 */
	template<typename Dtype>
	__global__ void SortInBlock(const int nthreads, const int block_num,
		int split_num, Dtype* energy_data, Dtype* index_data){
		CUDA_KERNEL_LOOP(index, nthreads){
			int next_block_num = block_num / split_num;
			int bw = index % next_block_num;
			int bh = (index / next_block_num) % next_block_num;
			int c = index / next_block_num/ next_block_num;
			int block_size = split_num * split_num;
			Dtype* sort_data = energy_data + c * block_num * block_num +
				bh * split_num * block_size + bw * block_size;
			Dtype* sort_index = index_data + c * block_num * block_num +
				bh * split_num * block_size + bw * block_size;
			bubble_sort<Dtype>(block_size, sort_data, sort_index);
		}
	}

	/*
	 * nthreads = num_ * channels_ * block_num * block_num
	 */
	template<typename Dtype>
	__global__ void ComputeOffset(int nthreads, int block_num, int split_num,
		int block_pixel_width, int block_pixel_height,
		const Dtype* index_data, Dtype* offset_w_data, Dtype* offset_h_data,
		Dtype* test_data){
		CUDA_KERNEL_LOOP(index, nthreads){
			// we only care about offset in sub-blocks
			int block_size = split_num * split_num;
			int sorted_iw = (index % block_size) % split_num;
			int sorted_ih = (index % block_size) / split_num;
			int id = int(index_data[index]);
			int c = index / block_num / block_num;
			int source_w =  id % block_num;
			int source_h = (id / block_num) % block_num;
			int source_iw = source_w % split_num;
			int source_ih = source_h % split_num;
			int offset_w = (sorted_iw - source_iw) * block_pixel_width;
			int offset_h = (sorted_ih - source_ih) * block_pixel_height;
			offset_w_data[c * block_num * block_num + id] = offset_w;
			offset_h_data[c * block_num * block_num + id] = offset_h;
			test_data[index] = int(index_data[index]);
		}
	}

  /*
   * @brief pass offsets from level p + 1 to p
   * nthreads = num_ * channels_ * block_num * block_num
   * merge next_level offset into current level offset
   */
	template<typename Dtype>
	__global__ void MergeOffset(int nthreads, int block_num, int split_num, 
		const Dtype* next_offset_w, const Dtype* next_offset_h,
		Dtype* curr_offset_w, Dtype* curr_offset_h){
		CUDA_KERNEL_LOOP(index, nthreads){
			int curr_bw = index % block_num;
			int next_bw = curr_bw / split_num;
			int curr_bh = (index / block_num) % block_num;
			int next_bh = curr_bh / split_num;
			int c = index / block_num / block_num;
			int next_block_num = block_num / split_num;
			int next_index = (c * next_block_num + next_bh) * next_block_num + next_bw;
			curr_offset_w[index] += next_offset_w[next_index];
			curr_offset_h[index] += next_offset_h[next_index];
		}
	}

	template<typename Dtype>
	void PatchRankLayer<Dtype>::SortBlock_gpu(){
		cudaStream_t* stream = new cudaStream_t[pyramid_height_];
		for (int i = 0; i < pyramid_height_; ++i){
			cudaStreamCreate(&stream[i]);
		}
		//sort 
		for (int p = 0; p < pyramid_height_; ++p){
			Dtype* energy_data = block_infos_[p]->mutable_gpu_data();
			Dtype* index_data = block_infos_[p]->mutable_gpu_diff();
			int block_num = block_infos_[p]->width();
			int N = num_ * channels_ * (block_num / split_num_) * 
				(block_num / split_num_);
			SortInBlock<Dtype><< < 1, CAFFE_GET_BLOCKS(N), 
				CAFFE_CUDA_NUM_THREADS, stream[p] >> >(
				N, block_num, split_num_, energy_data, index_data);
			CUDA_POST_KERNEL_CHECK;
//			ostringstream oss;
//			oss << p;
//			block_infos_[p]->ToTxt("block_info_sort_" + oss.str(), true);
		}
		for (int i = 0; i < pyramid_height_; ++i){
			cudaStreamDestroy(stream[i]);
		}
	}

	template<typename Dtype>
	void PatchRankLayer<Dtype>::ComputeLevelOffset_gpu(){
		cudaStream_t* stream = new cudaStream_t[pyramid_height_];
		//lunch multi-kernel to compute offset
		for (int i = 0; i < pyramid_height_; ++i){
			cudaStreamCreate(&stream[i]);
		}
		Dtype* test_data = test_data_.mutable_gpu_data();
		for (int p = 0; p < pyramid_height_; ++p){
			//offset of level p
			int count = block_offsets_[p]->count();
			int block_num = block_offsets_[p]->width();
			int block_pixel_width = num_unit_block_ / block_num * unit_block_width_;
			int block_pixel_height = num_unit_block_ / block_num * unit_block_height_;
			const Dtype* index_data = block_infos_[p]->gpu_diff();
			Dtype* offset_w_data = block_offsets_[p]->mutable_gpu_data();
			Dtype* offset_h_data = block_offsets_[p]->mutable_gpu_diff();
			ComputeOffset<Dtype> << <1, CAFFE_GET_BLOCKS(count),
				CAFFE_CUDA_NUM_THREADS, stream[p]>> >(count, block_num, split_num_,
				block_pixel_width, block_pixel_height, index_data, offset_w_data, 
				offset_h_data, test_data);
			CUDA_POST_KERNEL_CHECK;
			ostringstream oss;
			oss << p;
			block_offsets_[p]->ToTxt("block_offset_" + oss.str(), true);
		}
		for (int i = 0; i < pyramid_height_; ++i){
			cudaStreamDestroy(stream[i]);
		}
	}

	template<typename Dtype>
	void PatchRankLayer<Dtype>::MergeOffset_gpu(){
		cudaStream_t* stream = new cudaStream_t[pyramid_height_];
		//lunch multi-kernel
		for (int i = 0; i < pyramid_height_; ++i){
			cudaStreamCreate(&stream[i]);
		}
		//backpropagate offsets from level p to level 1
		for (int p = pyramid_height_ - 2; p >= 0; --p){
			//offset_{p} += offset_{p + 1}
			int count = block_offsets_[p]->count();
			int block_num = block_offsets_[p]->width();
			const Dtype* next_offset_w = block_offsets_[p + 1]->gpu_data();
			const Dtype* next_offset_h = block_offsets_[p + 1]->gpu_diff();
			Dtype* curr_offset_w = block_offsets_[p]->mutable_gpu_data();
			Dtype* curr_offset_h = block_offsets_[p]->mutable_gpu_diff();
			MergeOffset<Dtype> << <1, CAFFE_GET_BLOCKS(count), 
				CAFFE_CUDA_NUM_THREADS, stream[p] >> >(
				count, block_num, split_num_, next_offset_w, next_offset_h,
				curr_offset_w, curr_offset_h);
			CUDA_POST_KERNEL_CHECK;
			ostringstream oss;
			oss << p;
			block_offsets_[p]->ToTxt("block_offset_merge_" + oss.str(), true);
		}
		for (int i = 0; i < pyramid_height_; ++i){
			cudaStreamDestroy(stream[i]);
		}
	}

	/*
	 * nthreads = num_ * channels_ * height * width
	 */
	template<typename Dtype>
	__global__ void PatchRankForward(const int nthreads,
		const int unit_block_height, const int unit_block_width, 
		const int height, const int width, 
		const int num_unit_block, const Dtype* bottom_data, const Dtype* offset_h_data,
		const Dtype* offset_w_data, Dtype* top_data){
		CUDA_KERNEL_LOOP(index, nthreads){
			int w = index % width;
			int h = (index / width) % height;
			int c = index / width / height;
			int block_id_h = h / unit_block_height;
			int block_id_w = w / unit_block_width;
			/*
			 * for pixels not in the sorted blocks
			 * we just copy them to the output
			 */
			if (w >= num_unit_block || h >= num_unit_block){
				top_data[index] = bottom_data[index];
			}
			else{
				int offset_h = static_cast<int>(offset_h_data[c * num_unit_block *
					num_unit_block + block_id_h * num_unit_block + block_id_w]);
				int offset_w = static_cast<int>(offset_w_data[c * num_unit_block *
					num_unit_block + block_id_h * num_unit_block + block_id_w]);
				int top_w = w + offset_w;
				int top_h = h + offset_h;
				top_data[c * height * width + top_h * width + top_w] =
					bottom_data[index];
			}
		}
	}

	template<typename Dtype>
	void PatchRankLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top){
		const Dtype* bottom_data = bottom[0]->gpu_data();
		const Dtype* offset_w_data = block_offsets_[0]->gpu_data();
		const Dtype* offset_h_data = block_offsets_[0]->gpu_diff();
		Dtype* top_data = top[0]->mutable_gpu_data();
		GetBlockEnergy_gpu(bottom);
		SortBlock_gpu();
		ComputeLevelOffset_gpu();
		MergeOffset_gpu();
		const int count = bottom[0]->count();
		const int height = bottom[0]->height();
		const int width = bottom[0]->width();
		PatchRankForward<Dtype> << < CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS >> >(
			count, unit_block_height_, unit_block_width_, height, 
			width, num_unit_block_,
			bottom_data, offset_h_data, offset_w_data, top_data);
		CUDA_POST_KERNEL_CHECK;
	}

	template<typename Dtype>
	__global__ void PatchRankBackward(const int nthreads,
		const int unit_block_height, const int unit_block_width, 
		const int height, const int width, 
		const int num_unit_block, const Dtype* top_diff, const Dtype* offset_h_data,
		const Dtype* offset_w_data, Dtype* bottom_diff, Dtype* test_data){
		CUDA_KERNEL_LOOP(index, nthreads){
			int w = index % width;
			int h = (index / width) % height;
			int c = index / width / height;
			int block_id_h = h / unit_block_height;
			int block_id_w = w / unit_block_width;
			/*
			 * for pixels not in the sorted blocks
			 * we just copy diffs to the bottom
			 */
			if (block_id_h >= num_unit_block || block_id_w >= num_unit_block){
				bottom_diff[index] = top_diff[index];
			}
			else{
				int offset_h = static_cast<int>(offset_h_data[c * num_unit_block *
					num_unit_block + block_id_h * num_unit_block + block_id_w]);
				int offset_w = static_cast<int>(offset_w_data[c * num_unit_block *
					num_unit_block + block_id_h * num_unit_block + block_id_w]);
				int top_w = w + offset_w;
				int top_h = h + offset_h;
				bottom_diff[index] = top_diff[c * height * width + top_h * width + top_w];
				test_data[index] = offset_w;
			}
		}
	}

	template<typename Dtype>
	void PatchRankLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down,
		const vector<Blob<Dtype>*>& bottom){
		Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
		block_offsets_[0]->ToTxt("offset_bak_0", true);
		const Dtype* offset_w_data = block_offsets_[0]->gpu_data();
		const Dtype* offset_h_data = block_offsets_[0]->gpu_diff();
		const Dtype* top_diff = top[0]->gpu_diff();
		const int count = bottom[0]->count();
		const int height = bottom[0]->height();
		const int width = bottom[0]->width();
		caffe_gpu_set<Dtype>(test_data_.count(), Dtype(0), 
			test_data_.mutable_gpu_data());
		test_data_.ToTxt("test_before");
		PatchRankBackward<Dtype> << < CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS >> >(
			count, unit_block_height_, unit_block_width_,  
			height, width, num_unit_block_,
			top_diff, offset_h_data, offset_w_data, 
			bottom_diff, test_data_.mutable_gpu_data());
		CUDA_POST_KERNEL_CHECK;
		test_data_.ToTxt("test_data");
	}

	INSTANTIATE_LAYER_GPU_FUNCS(PatchRankLayer);

} // namespace caffe
