/********************************************************************************
** Copyright(c) 2016 USTC Reserved.
** auth: Xu Shen
** mail: shenxu@mail.ustc.edu.cn
** date: 2016/04/29
** desc: PatchRankLayer(GPU)
*********************************************************************************/
#include "caffe/layers/patch_rank_layer.hpp"

namespace caffe{

	/*
	 * nthreads: total number of unit blocks
	 * num_ * channels_ * num_unit_block_ * num_unit_block_
	 */
	template<typename Dtype>
	__global__ void ComputeBlockEnergyL1(const int nthreads,
		const int height, const int width, const int unit_block_height,
		const int unit_block_width, const int num_unit_block, const Dtype* bottom_data,
		Dtype* energy_data){
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
			energy_data[(c * num_unit_block + bh) * num_unit_block
				+ bw] = sum;
		}
	}

	/*
	 * nthreads: total number of unit blocks
	 * num_ * channels_ * num_unit_block_ * num_unit_block_
	 */
	template<typename Dtype>
	__global__ void ComputeBlockEnergyL2(const int nthreads,
		const int height, const int width, const int unit_block_height,
		const int unit_block_width, const int num_unit_block, const Dtype* bottom_data,
		Dtype* energy_data){
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
			energy_data[(c * num_unit_block + bh) * num_unit_block
				+ bw] = sum;
		}
	}

	template<typename Dtype>
	void PatchRankLayer<Dtype>::GetBlockEnergy_gpu(const vector<Blob<Dtype>*>& bottom){
		const Dtype* bottom_data = bottom[0]->gpu_data();
		Dtype* energy_data = block_energies_.mutable_gpu_data();
		const int width = bottom[0]->width();
		const int height = bottom[0]->height();
		const int total_blocks = num_ * channels_ * num_unit_block_ * num_unit_block_;
		switch (energy_type_){
		case PatchRankParameter_EnergyType_L1:
			ComputeBlockEnergyL1<Dtype> << <CAFFE_GET_BLOCKS(total_blocks), CAFFE_CUDA_NUM_THREADS >> >(
				total_blocks, height, width, unit_block_height_, unit_block_width_,
				num_unit_block_, bottom_data, energy_data);
			CUDA_POST_KERNEL_CHECK;
			break;
		case PatchRankParameter_EnergyType_L2:
			ComputeBlockEnergyL2<Dtype> << <CAFFE_GET_BLOCKS(total_blocks), CAFFE_CUDA_NUM_THREADS >> >(
				total_blocks, height, width, unit_block_height_, unit_block_width_,
				num_unit_block_, bottom_data, energy_data);
			CUDA_POST_KERNEL_CHECK;
			break;
		default:
			LOG(FATAL) << "Unkown energy type.";
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
   * nthreads = num_ * channels_ * outer_num * outer_num
   * each thread will sort inside each sub-blocks 
   */
	template<typename Dtype>
	__global__ void ComputeBlockOffset(int nthreads, int split_num, 
		int outer_dim, int outer_num, int num_unit_block,
		int unit_block_height, int unit_block_width,
		const Dtype* energy_data, Dtype* offset_h_data, Dtype* offset_w_data,
		Dtype* test_data){
		CUDA_KERNEL_LOOP(index, nthreads){
			const int num_part = split_num * split_num;
			// 16 KB limitation of local memory for each thread
			Dtype* block_energy = (Dtype*)malloc(num_part * sizeof(Dtype));
			Dtype* indexes = (Dtype*)malloc(num_part * sizeof(Dtype));
			int ow = index % outer_num;
			int oh = (index / outer_num) % outer_num;
			int c = index / outer_num / outer_num;
			int inner_dim = outer_dim / split_num;
			test_data[2 * num_part] = Dtype(inner_dim);
			test_data[2 * num_part + 1] = Dtype(outer_dim);
			int ooffset = c * num_unit_block * num_unit_block + oh *
				outer_dim * num_unit_block + ow * outer_dim;
			for (int ih = 0; ih < split_num; ++ih){
				for (int iw = 0; iw < split_num; ++iw){
					Dtype sum = 0;
					int ioffset = ih * inner_dim * num_unit_block + iw * inner_dim;
					for (int h = 0; h < inner_dim; ++h){
						for (int w = 0; w < inner_dim; ++w){
							sum += energy_data[ooffset + ioffset + h * num_unit_block + w];
						}
					}
					block_energy[ih * split_num + iw] = sum;
					indexes[ih * split_num + iw] = Dtype(ih * split_num + iw);
				}
			}
			//sort
			bubble_sort<Dtype>(num_part, block_energy, indexes);
			for (int i = 0; i < num_part; ++i){
				test_data[i] = block_energy[i];
				test_data[i + num_part] = indexes[i];
			}
			//offset
			for (int b = 0; b < num_part; ++b){
				int sorted_bw = b % split_num;
				int sorted_bh = b / split_num;
				int source_bw = int(indexes[b]) % split_num;
				int source_bh = int(indexes[b]) / split_num;
				//pixel offset in bottom feature map
				int offset_h = (sorted_bh - source_bh) * inner_dim * unit_block_height;
				int offset_w = (sorted_bw - source_bw) * inner_dim * unit_block_width;
				if (offset_h == 0 && offset_w == 0){
					continue;
				}
				//update offset of unit blocks
				for (int h = 0; h < inner_dim; ++h){
					for (int w = 0; w < inner_dim; ++w){
						//accumulated accross different pyramid levels
						offset_w_data[ooffset + source_bh * inner_dim * num_unit_block
							+ source_bw * inner_dim + h * num_unit_block + w] += offset_w;
						offset_h_data[ooffset + source_bh * inner_dim * num_unit_block
							+ source_bw * inner_dim + h * num_unit_block + w] += offset_h;
					}
				}
			}
			free(block_energy);
			free(indexes);
			__syncthreads();
		}
	}

	template<typename Dtype>
	void PatchRankLayer<Dtype>::GetBlockOffset_gpu(){
		const Dtype* energy_data = block_energies_.gpu_data();
		Dtype* offset_h_data = block_offsets_.mutable_gpu_diff();
		Dtype* offset_w_data = block_offsets_.mutable_gpu_data();
		//clear
		caffe_gpu_set<Dtype>(block_offsets_.count(), Dtype(0), offset_h_data);
		caffe_gpu_set<Dtype>(block_offsets_.count(), Dtype(0), offset_w_data);
		Dtype* test_data = block_energies_.mutable_gpu_diff();
		cudaStream_t* stream = new cudaStream_t[pyramid_height_];
		//lunch multi-kernel
		for (int i = 0; i < pyramid_height_; ++i){
			cudaStreamCreate(&stream[i]);
		}
		for (int p = 0; p < pyramid_height_; ++p){
			int outer_dim = pow(split_num_, pyramid_height_ - p);
			int outer_num = pow(split_num_, p);
			int nthreads = num_ * channels_ * outer_num * outer_num;
			ComputeBlockOffset<Dtype> << <1, CAFFE_GET_BLOCKS(nthreads), 
				CAFFE_CUDA_NUM_THREADS, stream[p] >> >(
				nthreads, split_num_, outer_dim, outer_num, num_unit_block_,
				unit_block_height_, unit_block_width_, energy_data, offset_h_data,
				offset_w_data, test_data);
			CUDA_POST_KERNEL_CHECK;
		}
		for (int i = 0; i < pyramid_height_; ++i){
			cudaStreamDestroy(stream[i]);
		}
		block_offsets_.ToTxt("offset_gpu",true);
		block_energies_.ToTxt("energy_gpu", true);
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
			if (w == num_unit_block || h == num_unit_block){
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
		const Dtype* offset_w_data = block_offsets_.gpu_data();
		const Dtype* offset_h_data = block_offsets_.gpu_diff();
		Dtype* top_data = top[0]->mutable_gpu_data();
		GetBlockEnergy_gpu(bottom);
		GetBlockOffset_gpu();
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
		const Dtype* offset_w_data, Dtype* bottom_diff){
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
			if (block_id_h == num_unit_block || block_id_w == num_unit_block){
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
			}
		}
	}

	template<typename Dtype>
	void PatchRankLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down,
		const vector<Blob<Dtype>*>& bottom){
		Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
		const Dtype* offset_w_data = block_offsets_.gpu_data();
		const Dtype* offset_h_data = block_offsets_.gpu_diff();
		const Dtype* top_diff = top[0]->gpu_diff();
		const int count = bottom[0]->count();
		const int height = bottom[0]->height();
		const int width = bottom[0]->width();
		caffe_copy<Dtype>(count, top_diff, bottom_diff);
		PatchRankBackward<Dtype> << < CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS >> >(
			count, unit_block_height_, unit_block_width_,  
			height, width, num_unit_block_,
			top_diff, offset_h_data, offset_w_data, bottom_diff);
		CUDA_POST_KERNEL_CHECK;
	}

	INSTANTIATE_LAYER_GPU_FUNCS(PatchRankLayer);

} // namespace caffe
