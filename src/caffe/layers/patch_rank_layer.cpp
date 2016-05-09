/********************************************************************************
** Copyright(c) 2016 USTC Reserved.
** auth: Xu Shen
** mail: shenxu@mail.ustc.edu.cn
** date: 2016/04/29
** desc: PatchRankLayer(CPU)
*********************************************************************************/
#include "caffe/layers/patch_rank_layer.hpp"

using std::pair;

namespace caffe{

	template<typename Dtype>
	bool descend_comp(const pair<Dtype, int>& left, const pair<Dtype, int>& right){
		return left.first > right.first;
	}

	template<typename Dtype>
	void PatchRankLayer<Dtype>::GetBlockEnergy_cpu(const vector<Blob<Dtype>*>& bottom){
		const Dtype* bottom_data = bottom[0]->cpu_data();
		Dtype* energy_data = this->block_infos_[0]->mutable_cpu_data();
		const int width = bottom[0]->width();
		for (int n = 0; n < num_; ++n){
			for (int c = 0; c < channels_; ++c){
				for (int bh = 0; bh < num_unit_block_; ++bh){
					for (int bw = 0; bw < num_unit_block_; ++bw){
						int start_w = bw * unit_block_width_;
						int start_h = bh * unit_block_height_;
						int end_w = (bw + 1) * unit_block_width_;
						int end_h = (bh + 1) * unit_block_height_;
						Dtype sum = 0;
						//get L1 or L2 norm of the block
						switch (energy_type_){
						case PatchRankParameter_EnergyType_L1:
							for (int h = start_h; h < end_h; ++h){
								for (int w = start_w; w < end_w; ++w){
									sum += abs(bottom_data[h * width + w]);
								}
							}
							break;
						case PatchRankParameter_EnergyType_L2:
							for (int h = start_h; h < end_h; ++h){
								for (int w = start_w; w < end_w; ++w){
									Dtype data = bottom_data[h * width + w];
									sum += data * data;
								}
							}
							break;
						default:
							LOG(FATAL) << "Unknown energy type";
							break;
						}
						energy_data[bh * num_unit_block_ + bw] = sum;
					}//for (int bw = 0; bw < num_unit_block_; ++bw)
				}//for (int bh = 0; bh < num_unit_block_; ++bh)
				energy_data += block_infos_[0]->offset(0, 1);
			}//for (int c = 0; c < channels_; ++c)
		}//for (int n = 0; n < num_; ++n)
	}

	/*
	 * map every thing to unit block mat
	 */
	template <typename Dtype>
	void PatchRankLayer<Dtype>::GetBlockOffset_cpu(){
		const Dtype* energy_data = this->block_infos_[0]->cpu_data();
		Dtype* offset_w_data = this->block_offsets_[0]->mutable_cpu_data();
		Dtype* offset_h_data = this->block_offsets_[0]->mutable_cpu_diff();
		//clear
		caffe_set<Dtype>(block_offsets_[0]->count(), Dtype(0.), offset_w_data);
		caffe_set<Dtype>(block_offsets_[0]->count(), Dtype(0.), offset_h_data);
		for (int n = 0; n < num_; ++n){
			for (int c = 0; c < channels_; ++c){
				for (int p = 0; p < pyramid_height_; ++p){
					//number of "big" blocks in this level
					const int outer_dim = pow(split_num_, pyramid_height_ - p);
					const int outer_num = pow(split_num_, p);
					for (int bh = 0; bh < outer_num; ++bh){
						for (int bw = 0; bw < outer_num; ++bw){
							//sort inside each sub-blocks 
							int inner_dim = outer_dim / split_num_;
							int ooffset = bh * num_unit_block_ * outer_dim
								+ bw * outer_dim;
							vector<std::pair<Dtype, int> > block_info;
							for (int ih = 0; ih < split_num_; ++ih){
								for (int iw = 0; iw < split_num_; ++iw){
									Dtype sum = 0;
									int ioffset = ih * inner_dim * num_unit_block_ 
										+ iw * inner_dim;
									for (int h = 0; h < inner_dim; ++h){
										for (int w = 0; w < inner_dim; ++w){
											sum += energy_data[ooffset + ioffset + h * num_unit_block_ + w];
										}
									}
									block_info.push_back(std::make_pair(sum,
										ih * split_num_ + iw));
								}
							}
							std::sort(block_info.begin(), block_info.end(), descend_comp<Dtype>);
							for (size_t b = 0; b < block_info.size(); ++b){
								int sorted_bw = b % split_num_;
								int sorted_bh = b / split_num_;
								int source_bw = block_info[b].second % split_num_;
								int source_bh = block_info[b].second / split_num_;
								//pixel offset in bottom feature map
								int offset_h = (sorted_bh - source_bh) * inner_dim * unit_block_height_;
								int offset_w = (sorted_bw - source_bw) * inner_dim * unit_block_width_;
								if (offset_h == 0 && offset_w == 0){
									continue;
								}
								//update offset of unit blocks
								for (int h = 0; h < inner_dim; ++h){
									for (int w = 0; w < inner_dim; ++w){
										//accumulated accross different pyramid levels
										offset_h_data[ooffset + source_bh * inner_dim * num_unit_block_
											+ source_bw * inner_dim + h * num_unit_block_ + w] += offset_h;
										offset_w_data[ooffset + source_bh * inner_dim * num_unit_block_
											+ source_bw * inner_dim + h * num_unit_block_ + w] += offset_w;
									}
								}
							}//for (size_t b = 0; b < block_info.size(); ++b)
						}//for (int bw = 0; bw < outer_block; ++bw)
					}//for (int bh = 0; bh < outer_block; ++bh)
				}//for (int p = 0; p < pyramid_height_; ++p)
				offset_h_data += block_offsets_[0]->offset(0, 1);
				offset_w_data += block_offsets_[0]->offset(0, 1);
				energy_data += block_infos_[0]->offset(0, 1);
			}//for (int c = 0; c < channels_; ++c)
		}//for (int n = 0; n < num_; ++n)
	}

	template<typename Dtype>
	void PatchRankLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top){
		pyramid_height_ = this->layer_param_.patch_rank_param().pyramid_height();
		CHECK_GT(pyramid_height_, 0);
		split_num_ = this->layer_param_.patch_rank_param().block_num();
		CHECK_GT(split_num_, 0);
		energy_type_ = this->layer_param_.patch_rank_param().energy_type();
		for (int i = 0; i < pyramid_height_; ++i){
			block_infos_.push_back(new Blob<Dtype>());
			block_offsets_.push_back(new Blob<Dtype>());
		}
	}

	template<typename Dtype>
	void PatchRankLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top){
		const int width = bottom[0]->width();
		const int height = bottom[0]->height();
		num_ = bottom[0]->num();
		channels_ = bottom[0]->channels();
		num_unit_block_ = pow(split_num_, pyramid_height_);
		unit_block_width_ = width / num_unit_block_;
		unit_block_height_ = height / num_unit_block_;
		CHECK_GE(unit_block_width_, 1) << "number of unit blocks should be less or "
			<< " equal than feature map width";
		CHECK_GE(unit_block_height_, 1)<< "number of unit blocks should be less or "
			<< " equal than feature map height";
		top[0]->ReshapeLike(*bottom[0]);
		//level 0 to level pyramid_height_ - 1
		for (int i = 0; i < pyramid_height_; ++i){
			int num_block = pow(split_num_, pyramid_height_ - i);
			block_infos_[i]->Reshape(num_, channels_, num_block, num_block);
			block_offsets_[i]->Reshape(num_, channels_, num_block, num_block);
		}
		test_data_.Reshape(num_, channels_, height, width);
	}

	template<typename Dtype>
	void PatchRankLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top){
		GetBlockEnergy_cpu(bottom);
		GetBlockOffset_cpu();
		const Dtype* offset_w_data = block_offsets_[0]->cpu_data();
		const Dtype* offset_h_data = block_offsets_[0]->cpu_diff();
		const Dtype* bottom_data = bottom[0]->cpu_data();
		Dtype* top_data = top[0]->mutable_cpu_data();
		const int height = bottom[0]->height();
		const int width = bottom[0]->width();
		for (int n = 0; n < num_; ++n){
			for (int c = 0; c < channels_; ++c){
				for (int h = 0; h < height; ++h){
					int block_id_h = h / unit_block_height_;
					for (int w = 0; w < width; ++w){
						int block_id_w = w / unit_block_width_;
						/*
						 * for pixels not in the sorted blocks
						 * we just copy them to the output
						 */
						if (block_id_h == num_unit_block_ || block_id_w == num_unit_block_){
							top_data[h * width + w] = bottom_data[h * width + w];
						}
						else{
							int offset_w = static_cast<int>(offset_w_data[block_id_h *
								num_unit_block_ + block_id_w]);
							int offset_h = static_cast<int>(offset_h_data[block_id_h *
								num_unit_block_ + block_id_w]);
							int top_w = w + offset_w;
							int top_h = h + offset_h;
							top_data[top_h * width + top_w] = bottom_data[h * width + w];
						}
					}
				}
				offset_w_data += block_offsets_[0]->offset(0, 1);
				offset_h_data += block_offsets_[0]->offset(0, 1);
				bottom_data += bottom[0]->offset(0, 1);
				top_data += top[0]->offset(0, 1);
			}//for (int c = 0; c < channels_; ++c)
		}//for (int n = 0; n < num_; ++n)
	}

	template<typename Dtype>
	void PatchRankLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down,
		const vector<Blob<Dtype>*>& bottom){
		const Dtype* offset_w_data = block_offsets_[0]->cpu_data();
		const Dtype* offset_h_data = block_offsets_[0]->cpu_diff();
		Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
		const Dtype* top_diff = top[0]->cpu_diff();
		const int height = bottom[0]->height();
		const int width = bottom[0]->width();
		for (int n = 0; n < num_; ++n){
			for (int c = 0; c < channels_; ++c){
				for (int h = 0; h < height; ++h){
					int block_id_h = h / unit_block_height_;
					for (int w = 0; w < width; ++w){
						int block_id_w = w / unit_block_width_;
						/*
						 * for pixels not in the sorted blocks
						 * we just copy diffs to the bottom
						 */
						if (block_id_h == num_unit_block_ || block_id_w == num_unit_block_){
							bottom_diff[h * width + w] = top_diff[h * width + w];
						}
						else{
							int offset_w = static_cast<int>(offset_w_data[block_id_h *
								num_unit_block_ + block_id_w]);
							int offset_h = static_cast<int>(offset_h_data[block_id_h *
								num_unit_block_ + block_id_w]);
							int top_w = w + offset_w;
							int top_h = h + offset_h;
							bottom_diff[h * width + w] = top_diff[top_h * width + top_w];
						}
					}
				}
				offset_w_data += block_offsets_[0]->offset(0, 1);
				offset_h_data += block_offsets_[0]->offset(0, 1);
				bottom_diff += bottom[0]->offset(0, 1);
				top_diff += top[0]->offset(0, 1);
			}//for (int c = 0; c < channels_; ++c)
		}//for (int n = 0; n < num_; ++n)
	}

#ifdef CPU_ONLY
	STUB_GPU(PatchRankLayer);
#endif

	INSTANTIATE_CLASS(PatchRankLayer);
	REGISTER_LAYER_CLASS(PatchRank);
} //namespace caffe
