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

void change_proto_shape(caffe::LayerParameter* layer, int idx, vector<int>& target_shape){
	CHECK_GT(layer->blobs_size(), idx);
	caffe::Blob<float> old_blob;
	old_blob.FromProto(layer->blobs(idx));
	caffe::Blob<float> new_blob(target_shape);
	// change blob infos
	BlobProto* blob = layer->mutable_blobs(idx);
	// why ToProto function does not set these parameters?
	blob->set_num(target_shape[0]);
	blob->set_channels(target_shape[1]);
	blob->set_height(target_shape[2]);
	blob->set_width(target_shape[3]);
	new_blob.ToProto(blob);
	LOG(ERROR) << "Layer " << layer->name().c_str() << "(param " << idx << "): " 
		<< old_blob.shape_string().c_str() << " -> " << new_blob.shape_string();
}

void reset_proto_data(caffe::LayerParameter* layer, int idx){
	CHECK_GT(layer->blobs_size(), idx);
	caffe::Blob<float> old_blob;
	old_blob.FromProto(layer->blobs(idx));
	caffe_set<float>(old_blob.count(), 0., old_blob.mutable_cpu_data());
	BlobProto* blob = layer->mutable_blobs(idx);
	// why ToProto function does not set these parameters?
	old_blob.ToProto(blob);
	LOG(ERROR) << "Reset Layer " << layer->name().c_str() << "(param " << idx << ").";
}

int main(int argc, char** argv) {
//	::google::InitGoogleLogging(argv[0]);
	FLAGS_alsologtostderr = 0;
	if (argc != 4)
	{
		cout << "usage: change_net_params.exe model_1 model_2 save_caffemodel"
			<< " new_model_name" << endl;
		return 0;
	}
	std::string alex_model_path(argv[1]);
	std::string vgg_model_path(argv[2]);
	std::string new_model_path = argv[3];
	

	//convert model
	NetParameter alex_net_param, vgg_net_param;
	LOG(INFO) << "Loading alex params from: " << alex_model_path << " ...";
	LOG(INFO) << "Loading vgg params from: " << vgg_model_path << " ...";
//	caffe::ReadNetParamsFromTextFileOrDie("VGG_ILSVRC_19_layers_deconv_train_val_vgg_alex_data.prototxt", 
//		&alex_net_param);
	caffe::ReadProtoFromBinaryFile(alex_model_path, &alex_net_param);
	caffe::ReadProtoFromBinaryFile(vgg_model_path, &vgg_net_param);
	LOG(INFO) << "number of layers in vgg: " << vgg_net_param.layers_size();
	LOG(INFO) << "number of layers in alex: " << alex_net_param.layers_size();
	NetParameter whole_net_param;
	for (int j = 0; j < vgg_net_param.layers_size(); ++j){
		caffe::V1LayerParameter* v1_layer_param = vgg_net_param.mutable_layers(j);
		caffe::LayerParameter* target_layer = whole_net_param.add_layer();
		caffe::UpgradeV1LayerParameter(*v1_layer_param, target_layer);
	}
	for (int j = 0; j < alex_net_param.layers_size(); ++j)
	{
		caffe::LayerParameter layer_param;
		caffe::V1LayerParameter* v1_layer_param = alex_net_param.mutable_layers(j);
		caffe::UpgradeV1LayerParameter(*v1_layer_param, &layer_param);
		const std::string name = layer_param.name();
		// set all bias blob data to 0s
		if (name == "conv1"){
			vector<int> bias_shape{ 1, 1, 1, 3 };
			change_proto_shape(&layer_param, 1, bias_shape);
		}
		else if (name == "fc6"){
			vector<int> bias_shape{ 1, 1, 1, 9216};
			change_proto_shape(&layer_param, 1, bias_shape);
		}
		else if (name == "conv5"){
			vector<int> bias_shape{ 1, 1, 1, 384 };
			change_proto_shape(&layer_param, 1, bias_shape);
		}
		else if (name == "conv4"){
			vector<int> bias_shape{ 1, 1, 1, 384};
			change_proto_shape(&layer_param, 1, bias_shape);
		}
		else if (name == "conv3"){
			vector<int> bias_shape{ 1, 1, 1, 256 };
			change_proto_shape(&layer_param, 1, bias_shape);
		}
		else if (name == "conv2"){
			vector<int> bias_shape{ 1, 1, 1, 96 };
			change_proto_shape(&layer_param, 1, bias_shape);
		}
		else if (layer_param.blobs_size() > 1){
			// reset bias blob
			reset_proto_data(&layer_param, 1);
		}
		layer_param.set_name(name + "_deconv");
		caffe::LayerParameter* target_layer = whole_net_param.add_layer();
		target_layer->CopyFrom(layer_param);
	}
	caffe::WriteProtoToBinaryFile(whole_net_param, new_model_path);
	cout << "# of layers: " << whole_net_param.layer_size() << "\n";
//	caffe::WriteProtoToTextFile(net_param, "vgg_txt.caffemodel");
	return 0;
}