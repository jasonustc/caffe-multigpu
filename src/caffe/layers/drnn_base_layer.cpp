#include <string>
#include <utility>
#include <vector>


#include "caffe/filler.hpp"
#include "caffe/layers/drnn_base_layer.hpp"

namespace caffe{
	template <typename Dtype>
	void DRNNBaseLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top){
		// num_seq_, #streams, Dim_
		// H_
		CHECK_EQ(3, bottom[0]->num_axes());
		// C_
		CHECK_EQ(3, bottom[1]->num_axes());
		// cont_
		// T_, #streams 
		CHECK_EQ(2, bottom[2]->num_axes());
		CHECK_EQ(bottom[0]->shape(0), bottom[2]->shape(0));
		CHECK_EQ(bottom[0]->shape(1), bottom[2]->shape(1));
		CHECK_EQ(bottom[0]->shape(), bottom[1]->shape());
		reverse_ = this->layer_param_.recurrent_param().reverse();
		LOG_IF(INFO, reverse_) << "Decode input sequence in reverse order";
		conditional_ = this->layer_param_.recurrent_param().conditional();
		LOG_IF(INFO, conditional_) << "Decode input is groundtruth input sequence";
		if (conditional_){
			//X_
			CHECK_EQ(3, bottom[3]->num_axes());
			CHECK_EQ(bottom[2]->shape(0), bottom[3]->shape(0));
			CHECK_EQ(bottom[2]->shape(1), bottom[3]->shape(1));
		}
		hidden_dim_ = this->GetHiddenDim();
		num_seq_ = bottom[0]->shape(0);
		T_ = bottom[2]->shape(0);

		//shapes of blobs
		const vector<int> x_shape{
			1,
			bottom[3]->shape(1),
			bottom[3]->shape(2)
		};
		const vector<int> h_shape{
			1,
			bottom[0]->shape(1),
			bottom[0]->shape(2)
		};
		const vector<int> cont_shape{
			1,
			bottom[2]->shape(1)
		};

		// setup slice_h_ layer
		// Top
		H0_.resize(num_seq_);
		for (int n = 0; t < num_seq_; ++n){
			H0_[n].reset(new Blob<Dtype>(h_shape));
		}
		// Layer
		const vector<Blob<Dtype>*> slice_h_bottom(1, bottom[0]);
		const vector<Blob<Dtype>*> slice_h_top(num_seq_, H0_[0].get());
		LayerParameter slice_param;
		slice_param.mutable_slice_param()->set_axis(0);
		slice_h_.reset(new SliceLayer<Dtype>(slice_param));
		slice_h_->SetUp(slice_h_bottom, slice_h_top);

		// setup slice_c_ layer
		// Top
		C0_.resize(num_seq_);
		for (int n = 0; n < num_seq_; ++n){
			C0_[n].reset(new Blob<Dtype>(h_shape));
		}
		// Layer
		const vector<Blob<Dtype>*> slice_c_bottom(1, bottom[1]);
		const vector<Blob<Dtype>*> slice_c_top(num_seq_, C0_[0].get());
		slice_c_.reset(new SliceLayer<Dtype>(slice_param));
		slice_c_->SetUp(slice_c_bottom, slice_c_top);

		// setup slice_x_ layer
		// Top
		if (conditional_){
			X_.resize(T_);
			for (int t = 0; t < T_; ++t){
				X_[t].reset(new Blob<Dtype>(x_shape));
			}
			// Layer
			const vector<Blob<Dtype>*> slice_x_bottom(1, bottom[3]);
			const vector<Blob<Dtype>*> slice_x_top(T_, X_[0].get());
			slice_x_.reset(new SliceLayer<Dtype>(slice_param));
			slice_x_->SetUp(slice_x_bottom, slice_x_top);
		}
		
		// setup concat_h_ layer
		// Top
		H_DEC_.resize(T_);
		for (int t = 0; t < T_; ++t){
			H_DEC_[t].reset(new Blob<Dtype>(h_shape));
		}
		decode_output_.resize(T_, H_DEC_[0].get());
		vector<Blob<Dtype>*> concat_h_dec_bottom(T_, H_DEC_[0].get());
		const vector<Blob<Dtype>*> concat_h_dec_top(1, top[0]);

		// Layer
		LayerParameter concat_param;
		concat_param.mutable_concat_param()->set_axis(0);
		concat_h_dec_.reset(new ConcatLayer<Dtype>(concat_param));
		concat_h_dec_->SetUp(concat_h_dec_bottom, concat_h_dec_top);

		// setup zero_blob_ for decode input
		if (conditional_){
			zero_blob_.reset(new Blob<Dtype>(x_shape));
		}
		else{
			zero_blob_.reset(new Blob<Dtype>(h_shape));
		}
		FillerParameter filler_param;
		filler_param.set_type("constant");
		filler_param.set_value(0);
		shared_ptr<Filler<Dtype> > filler(GetFiller<Dtype>(filler_param));
		filler->Fill(zero_blob_.get());
	}

	template <typename Dtype>
	void DRNNBaseLayer<Dtype>::InferSeqLengths(Blob<Dtype>* cont){
		int seq_begin(0);
		const Dtype* cont_data = cont->cpu_data();
		const int count = cont->count();
		for (size_t s = 1; s < count; ++s){
			if (cont_data[s] == 0){
				seq_lens_.push_back(s - seq_begin);
				seq_begin = s;
			}
		}
		seq_lens_.push_back(count - seq_begin);
	}

	template <typename Dtype>
	void DRNNBaseLayer<Dtype>::ReorderDecodeOutpout(const Blob<Dtype>* cont){
		const Dtype* cont_data = cont->cpu_data();
		int seq_id = -1, seq_begin_id = 0;
		for (int t = 0; t < T_; ++t){
			//cont
			if (cont_data[t] == 0){
				seq_id++;
				seq_begin_id += seq_id > 0 ? seq_lens_[seq_id - 1] : 0;
			}
			if (reverse_){
				int curr_seq_len = seq_lens_[seq_id];
				H_DEC_[t] = decode_output_[seq_begin_id + 
					curr_seq_len - (t - seq_begin_id) - 1];
			}
			else{
				H_DEC_[t] = decode_output_[t];
			}
		}
	}

	template <typename Dtype>
	void DRNNBaseLayer<Dtype>::ReorderDecodeInput(const vector<Blob<Dtype>*>& bottom){
		int seq_id(-1), seq_begin_id(0);
		const Dtype* cont_data = bottom[2]->cpu_data();
		// NOTE: for the sake of backpropagation, we need to use split layer for 
		// data/output reuse
		if (conditional_){
			decode_input_.resize(T_, X_[0].get());
			for (int t = 0; t < T_; ++t){
				//cont
				if (cont_data[t] == 0){
					seq_id++;
					seq_begin_id += seq_id > 0 ? seq_lens_[seq_id - 1] : 0;
				}
				if (reverse_){
					//0, x_{T}, x_{T-1}, x_{T-2}, ...
					int curr_seq_len = seq_lens_[seq_id];
					decode_input_[t] = cont_data[t] == 0 ? zero_blob_ :
						X_[seq_begin_id + curr_seq_len - (t - seq_begin_id)] ;
				}
				else{
					//0, x_1, x_{2}, x_{3}, ...
					decode_input_[t] = cont_data[t] == 0 ? zero_blob_ : X_[t - 1];
				}
			}
		}
		else{
			decode_input_.resize(T_, H_DEC_[0].get());
			for (int t = 0; t < T_; ++t){
				//NOTE: maybe need split_layer_ here
				decode_input_[t] = cont_data[t] == 0 ? zero_blob_ : H_DEC_[t - 1];
			}
		}
	}

	template <typename Dtype>
	void DRNNBaseLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top){
		vector<int> top_shape{
			T_,
			bottom[0]->shape(1),
			bottom[0]->shape(2)
		};
		top[0]->Reshape(top_shape);
	}

	template <typename Dtype>
	void DRNNBaseLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top){
		this->ShareWeight();
		this->InferSeqLengths(bottom[2]);
		// 1. slice_h_
		const vector<Blob<Dtype>*> slice_h_bottom(1, bottom[0]);
		vector<Blob<Dtype>*> slice_h_top(num_seq_, NULL);
		for (int s = 0; s < num_seq_; ++s){
			slice_h_top[s] = H_[s].get();
		}
		slice_h_->Forward(slice_h_bottom, slice_h_top);

		// 2. slice_c_ 
		const vector<Blob<Dtype>*> slice_c_bottom(1, bottom[1]);
		vector<Blob<Dtype>*> slice_c_top(num_seq_, NULL);
		for (int s = 0; s < num_seq_; ++s){
			slice_c_top[s] = C_[s].get();
		}
		slice_c_->Forward(slice_c_bottom, slice_c_top);

		// 3. slice_x_ if needed
		if (conditional_){
			const vector<Blob<Dtype>*> slice_x_bottom(1, bottom[2]);
			vector<Blob<Dtype>*> slice_x_top(T_, NULL);
			for (int t = 0; t < T_; ++t){
				slice_x_top[t] = X_[t].get();
			}
		}

		// 4. reorder decode input
		this->ReorderDecodeInput(bottom);

		// 5. for all sequence, run decode lstm.
		for (int t = 0; t < T_; t++){
			this->RecurrentForward(t);
		}
		
		// 6. reorder decode output
		this->ReorderDecodeOutpout(bottom[2]);

		// 7. concat h_dec_
		const vector<Blob<Dtype>*> concat_h_dec_bottom(T_, NULL);
		for (int t = 0; t < T_; ++t){
			concat_h_dec_bottom[t] = H_DEC_[t].get();
		}
		const vector<Blob<Dtype>*> concat_h_dec_top(1, top[0]);
		concat_h_dec_->Forward(concat_h_dec_bottom, concat_h_dec_top);
	}

	template <typename Dtype>
	void DRNNBaseLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom){
		this->ShareWeight();

		// 7. concat h_dec_
		vector<Blob<Dtype>*> concat_h_dec_bottom(T_, NULL);
		for (int t = 0; t < T_; ++t){
			concat_h_dec_bottom[t] = H_DEC_[t].get();
		}
		const vector<Blob<Dtype>*> concat_h_dec_top(1, top[0]);
		concat_h_dec_->Backward(concat_h_dec_top,
			vector<bool>(T_, true),
			concat_ht_bottom);

		// 5. for all sequence, run decode LSTM
		for (int t = T_ - 1; t >= 0; --t){
			this->RecurrentBackward(t);
		}

		// 3. slice_x_ if needed
		if (conditional_){
			const vector<Blob<Dtype>*> slice_x_bottom(1, bottom[3]);
			vector<Blob<Dtype>*> slice_x_top(T_, NULL);
			for (int t = 0; t < T_; ++t){
				slice_x_top[t] = X_[t].get();
			}
			slice_x_->Backward(slice_x_top,
				vector<bool>(T_, true),
				slice_x_bottom);
		}

		// 2. slice_c_
		const vector<Blob<Dtype>*> slice_c_bottom(1, bottom[1]);
		vector<Blob<Dtype>*> slice_c_top(num_seq_, NULL);
		for (int n = 0; n < num_seq_; ++n){
			slice_c_top[n] = C0_[n];
		}
		slice_c_->Backward(slice_c_top,
			vector<bool>(num_seq_, true),
			slice_c_bottom);
		
		// 1. slice_h_
		const vector<Blob<Dtype>*> slice_h_bottom(1, bottom[0]);
		vector<Blob<Dtype>*> slice_h_top(num_seq_, NULL);
		for (int n = 0; n < num_seq_; ++n){
			slice_h_top[n] = H0_[n].get();
		}
		slice_h_->Backward(slice_h_top,
			vector<bool>(num_seq_, true),
			slice_h_bottom);
	}

	INSTANTIATE_CLASS(DRNNBaseLayer);
} // namespace caffe
