#include <glog/logging.h>

#include <cstring>
#include <map>
#include <string>
#include <vector>
#include <fstream>
#include <iostream> // NOLINT(readability/streams)

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
	if (argc != 4)
	{
		cout << "usage: merge_caffemodels.exe model1[,model2,...] suffix1[,suffix2,...,]"
			<< " save_model_name" << endl;
		return 0;
	}
	std::fstream out("model_file", std::ios::out | std::ios::trunc);
	std::string model_paths = argv[1];
	std::string suffixes = argv[2];
	std::vector<std::string> model_path;
	std::vector<std::string> suffix;
	boost::split(model_path, model_paths, boost::is_any_of(","));
	boost::split(suffix, suffixes, boost::is_any_of(","));
	CHECK_EQ(model_path.size(), suffix.size());
	std::string new_model_path = argv[3];

	//convert model
	NetParameter merged_param;
	for (size_t i = 0; i < model_path.size(); ++i){
		NetParameter param;
		LOG(INFO) << "Loading params from: " << model_path[i] << "...\n";
		caffe::ReadProtoFromBinaryFile(model_path[i], &param);
		//merge from param
		for (int j = 0; j < param.layer_size(); ++j)
		{
			caffe::LayerParameter* old_layer = param.mutable_layer(j);
			const std::string old_name = old_layer->name();
			std::string new_name = old_name + suffix[i];
			old_layer->set_name(new_name);
			caffe::LayerParameter* layer = merged_param.add_layer();
			layer->CopyFrom(*old_layer);
			LOG(INFO) << "Merged layer: " <<  layer->name() << "\n";
			out << layer->name() << "\n";
		}
	}
	out.close();
	caffe::WriteProtoToBinaryFile(merged_param, new_model_path);
	return 0;
}