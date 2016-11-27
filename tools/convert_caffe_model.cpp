#include <glog/logging.h>

#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "boost/algorithm/string.hpp"
#include "caffe/caffe.hpp"
#include "caffe/util/upgrade_proto.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/blob.hpp"

using std::cout;
using std::endl;
using std::string;
using std::vector;

using namespace caffe;

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
//	::google::InitGoogleLogging(argv[0]);
	FLAGS_alsologtostderr = 0;
	if (argc != 3)
	{
		cout << "usage: merge_caffemodels.exe bvlc_alexnet.caffemodel"
			<< " save_model_name" << endl;
		return 0;
	}
	std::string model_path(argv[1]);
	std::string new_model_path = argv[2];

	//convert model
	NetParameter param;
	LOG(INFO) << "Loading params from: " << model_path << "...";
	caffe::ReadProtoFromBinaryFile(model_path, &param);
	LOG(INFO) << "number of layers: " << param.layers_size();
	for (int j = 0; j < param.layers_size(); ++j)
	{
		caffe::V1LayerParameter* layer = param.mutable_layers(j);
		const std::string name = layer->name();
		// change the weight blob to 6-channels
		// to enable the model to deal with 6 channels input
		if (name == "conv1"){
			// usually the data type for caffe is float
			// duplicate the first 3 channels data to 
			// second 3 channels data
			caffe::Blob<float> weight_blob;
			weight_blob.FromProto(layer->blobs(0));
			vector<int> weight_shape = weight_blob.shape();
			weight_shape[1] = 6;
			const float* old_weight_data = weight_blob.cpu_data();
			caffe::Blob<float> new_weight_blob(weight_shape);
			float* new_weight_data = new_weight_blob.mutable_cpu_data();
			const int count = weight_blob.count();
			caffe_copy<float>(count, old_weight_data, new_weight_data);
			caffe_copy<float>(count, old_weight_data, new_weight_data + count);
			// change blob infos
			BlobProto* blob = layer->mutable_blobs(0);
			new_weight_blob.ToProto(blob);
			cout << blob->data(count) << "  " << blob->data(count + 100) << "\n";
			cout << blob->data(count + 1000) << "  " << blob->data(count + 2000) << "\n";
			break;
		}
	}
	caffe::WriteProtoToBinaryFile(param, new_model_path);
	return 0;
}