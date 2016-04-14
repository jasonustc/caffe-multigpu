
/********************************************************************************
** Copyright(c) 2015 USTC & MSRA All Rights Reserved.
** auth: Zhaofan Qiu
** mail: zhaofanqiu@gmail.com
** date: 2015/9/11
** desc: convert_c3d_model_and_mean tool
*********************************************************************************/

#include <glog/logging.h>

#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "boost/algorithm/string.hpp"
#include "caffe/caffe.hpp"
#include "caffe/util/upgrade_proto.hpp"

using std::cout;
using std::endl;
using std::string;
using std::vector;

using caffe::BlobProto;
using caffe::NetParameter;

void resize_blob_proto(BlobProto* proto, vector<int> shape)
{
	LOG(INFO) << "Convert layer";
	int count = 1;
	for (int i = 0; i < shape.size(); i++)
	{
		count *= shape[i];
	}
	CHECK_EQ(count, proto->data_size());
	proto->clear_shape();
	for (int i = 0; i < shape.size(); i++)
	{
		proto->mutable_shape()->add_dim(shape[i]);
	}
}

void process_fc6(BlobProto* proto)
{
	LOG(INFO) << "process_fc6";
	proto->set_num(4096);
	proto->set_channels(512);
	proto->set_height(7);
	proto->set_width(7);
//	proto->clear_shape();
//	proto->mutable_shape()->add_dim(4096);
//	proto->mutable_shape()->add_dim(512);
//	proto->mutable_shape()->add_dim(7);
//	proto->mutable_shape()->add_dim(7);
}

void process_fc7(BlobProto* proto)
{
	LOG(INFO) << "process_fc7";
	proto->set_num(4096);
	proto->set_channels(4096);
	proto->set_height(1);
	proto->set_width(1);

//	proto->clear_shape();
//	proto->mutable_shape()->add_dim(4096);
//	proto->mutable_shape()->add_dim(4096);
//	proto->mutable_shape()->add_dim(1);
//	proto->mutable_shape()->add_dim(1);
}

void process_fc8(BlobProto* proto)
{
	LOG(INFO) << "process_fc8";
	proto->set_num(1000);
	proto->set_channels(4096);
	proto->set_height(1);
	proto->set_width(1);
//	proto->clear_shape();
//	proto->mutable_shape()->add_dim(1000);
//	proto->mutable_shape()->add_dim(4096);
//	proto->mutable_shape()->add_dim(1);
//	proto->mutable_shape()->add_dim(1);
}

int main(int argc, char** argv) {
	FLAGS_alsologtostderr = 0;
	if (argc != 3)
	{
		cout << "usage: convert_vgg_model_full_convolutional.exe vgg_16_layers_model save_model_name" << endl;
		return 0;
	}
	std::string model_path = argv[1];
	std::string new_model_path = argv[2];

	//convert model
	NetParameter param;
	caffe::ReadProtoFromBinaryFile(model_path, &param);
	cout << param.layers_size() << "\n";
	for (int i = 0; i < param.layers_size(); ++i) 
	{
		caffe::V1LayerParameter* layer = param.mutable_layers(i);
		if (layer->name() == "fc6")
		{
			process_fc6(layer->mutable_blobs(0));
		}
		if (layer->name() == "fc7")
		{
			process_fc7(layer->mutable_blobs(0));
		}
		if (layer->name() == "fc8"){
			process_fc8(layer->mutable_blobs(0));
		}
	}
	caffe::WriteProtoToBinaryFile(param, new_model_path);
	return 0;
}