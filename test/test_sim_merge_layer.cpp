#include <cstring>
#include <vector>

#include "gtest/gtest.h"
#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/test/test_gradient_check_util.hpp"
#include "caffe/test/test_caffe_main.hpp"
#include "caffe/util/sim_merge.hpp"

namespace caffe{
	template <typename Dtype>
	class SimMergeLayerTest{
	public:
		SimMergeLayerTest() : input_map_(new Blob<Dtype>()){
			this->SetUp();
		}
		~SimMergeLayerTest(){ delete input_map_; }
		void TestSetUp(Caffe::Brew mode){
			Caffe::set_mode(mode);
			test_net_->Forward();
			Blob<Dtype>* out = CHECK_NOTNULL(test_net_->blob_by_name("ip_out").get());
			CHECK_EQ(out->shape(0), 2);
			CHECK_EQ(out->shape(1), 5);
		}
		void TestForward(Caffe::Brew mode){
			Caffe::set_mode(mode);
			test_net_->MergeAndRefreshWeights();
			test_net_->Forward();
		}
		
		void TestBackward(Caffe::Brew mode){
			Caffe::set_mode(mode);
			test_net_->ForwardBackward();
		}

	private:
		void SetUp(){
			//setup pointers to inputs
			//setup input data
			vector<int> x_shape;
			x_shape.push_back(2);
			x_shape.push_back(3);
			x_shape.push_back(2);
			x_shape.push_back(2);
			input_map_ = new Blob<Dtype>(x_shape);
			FillerParameter filler_param;
			filler_param.set_value(1);
			ConstantFiller<Dtype> filler(filler_param);
			filler.Fill(input_map_);
			Dtype* x_data = input_map_->mutable_cpu_data();
			x_data[0] = (Dtype)0;
			x_data[1] = (Dtype)1;
			x_data[2] = (Dtype)-1;
			x_data[5] = (Dtype)3;
			x_data[7] = (Dtype)0.5;
			x_data[9] = (Dtype)2;
			x_data[11] = (Dtype)0.1;
			x_data[12] = (Dtype)0;
			x_data[13] = (Dtype)1;
			x_data[14] = (Dtype)-1;
			//set layer parameter
			layer_param_.mutable_convolution_param()->mutable_weight_filler()->set_type("gaussian");
			layer_param_.mutable_convolution_param()->mutable_weight_filler()->set_std(0.01);
			layer_param_.mutable_convolution_param()->mutable_bias_filler()->set_type("constant");
			layer_param_.mutable_convolution_param()->mutable_bias_filler()->set_value(0.1);
			layer_param_.mutable_convolution_param()->set_num_output(4);
			layer_param_.mutable_convolution_param()->add_kernel_size(2);

			//build a test net
			//TODO: why this way do not work?
			/*
			net_param.add_input("x");
			BlobShape input_shape;
			input_shape.add_dim(2);
			input_shape.add_dim(3);
			input_shape.add_dim(2);
			input_shape.add_dim(2);
			net_param.add_input_shape()->CopyFrom(input_shape);
			*/

			//build net input, the top blobs need to be assigned manually
			NetParameter net_param;
			LayerParameter* input_param = net_param.add_layer();
			input_param->set_name("data");
			input_param->set_type("Input");
			input_param->add_top("data");
			BlobShape input_shape;
			input_shape.add_dim(2);
			input_shape.add_dim(3);
			input_shape.add_dim(2);
			input_shape.add_dim(2);
			input_param->mutable_input_param()->add_shape()->CopyFrom(input_shape);


			LayerParameter* conv_param = net_param.add_layer();
			conv_param->CopyFrom(layer_param_);
			conv_param->add_bottom("data");
			conv_param->add_top("conv");
			conv_param->set_type("Convolution");
			conv_param->add_param()->set_name("conv_weight");
			conv_param->add_param()->set_name("conv_bias");
			conv_param->set_name("conv");

			LayerParameter* ip_param = net_param.add_layer();
			ip_param->mutable_inner_product_param()->mutable_weight_filler()->set_type("gaussian");
			ip_param->mutable_inner_product_param()->mutable_weight_filler()->set_std(0.1);
			ip_param->mutable_inner_product_param()->mutable_bias_filler()->set_type("constant");
			ip_param->mutable_inner_product_param()->mutable_bias_filler()->set_value(0);
			ip_param->mutable_inner_product_param()->set_num_output(5);
			ip_param->add_bottom("conv");
			ip_param->add_top("ip_out");
			ip_param->set_type("InnerProduct");
			ip_param->set_name("ip_out");
			ip_param->add_param()->set_name("ip_weight");
			ip_param->add_param()->set_name("ip_bias");

			MergeParamSpec* merge_param = net_param.add_merge_param();
			merge_param->set_axis(1);
			merge_param->set_name("conv_weight");
			merge_param->set_prop(0.1);
			merge_param->mutable_filler()->set_type("gaussian");
			merge_param->mutable_filler()->set_std(0.1);
			merge_param->set_hard(true);

			net_param.set_debug_info(true);
			test_net_.reset(new Net<Dtype>(net_param));
			Blob<Dtype>* net_input_ = CHECK_NOTNULL(test_net_->blob_by_name("data").get());
			net_input_->ReshapeLike(*input_map_);
			net_input_->ShareData(*input_map_);
			net_input_->ShareDiff(*input_map_);
		}

		LayerParameter layer_param_;

		Blob<Dtype>* input_map_;

		vector<bool> propagate_down_;

		shared_ptr<Net<Dtype> > test_net_;
	};
}

int main(int argc, char** argv){
	::google::InitGoogleLogging(*argv);
//	FLAGS_alsologtostderr = true;
	FLAGS_logtostderr = true;
	caffe::SimMergeLayerTest<float> test;
	test.TestSetUp(caffe::Caffe::CPU);
//	test.TestForward(caffe::Caffe::CPU);
//	test.TestBackward(caffe::Caffe::CPU);
	test.TestForward(caffe::Caffe::GPU);
//	test.TestBackward(caffe::Caffe::GPU);
	return 0;
}
