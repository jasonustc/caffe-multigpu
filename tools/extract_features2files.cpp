#include <cstring>
#include <cstdlib>
#include <vector>

#include <fstream>
#include <iostream>

#include "boost/algorithm/string.hpp"
#include "google/protobuf/text_format.h"
#include "caffe/util/db.hpp"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/net.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"

using namespace caffe;  // NOLINT(build/namespaces)

#ifdef _MSC_VER
DEFINE_string(log_dir, "log",
	"Optional; directory to save log file");
DEFINE_string(log_name, "caffe.log.",
	"Optional; name prefix of the log file");
#endif

DEFINE_bool(sqrt, false,
	"Optional; if we need to do root square on extracted features.");
DEFINE_bool(l2_norm, false,
	"Optional; if we need to do L2 normalization on extracted features.");

template<typename Dtype>
int feature_extraction_pipeline(int argc, char** argv);

int main(int argc, char** argv) {
#ifdef _MSC_VER
	//set log file directory
	gflags::ParseCommandLineFlags(&argc, &argv, true);
	boost::filesystem::create_directory(FLAGS_log_dir);
	string log_dest = FLAGS_log_dir + "/";
	::google::SetLogDestination(0, log_dest.c_str());
	::google::SetLogFilenameExtension(FLAGS_log_name.c_str());
#endif
	::google::InitGoogleLogging(argv[0]);
	::google::SetStderrLogging(0);
	//return feature_extraction_pipeline<float>(argc, argv);
	return feature_extraction_pipeline<double>(argc, argv);
}

template<typename Dtype>
void L2Normalize(const int dim, Dtype* feat_data){
	Dtype l2_norm = caffe_cpu_dot<Dtype>(dim, feat_data, feat_data);
	caffe_scal<Dtype>(dim, Dtype(1. / l2_norm), feat_data);
}

template<typename Dtype>
int feature_extraction_pipeline(int argc, char** argv) {
	const int num_required_args = 6;
	if (argc < num_required_args) {
		LOG(ERROR)<<
			"This program takes in a trained network and an input data layer, and then"
			" extract features of the input data produced by the net.\n"
			"Usage: extract_features  [-log_dir,-log_name,--sqrt,--l2_norm]"
			" pretrained_net_param"
			"  feature_extraction_proto_file  extract_feature_blob_name1[,name2,...]"
			"  save_feature_file_name1[,name2,...]  num_mini_batches  [CPU/GPU]"
			"  [DEVICE_ID=0]\n"
			"Note: you can extract multiple features in one pass by specifying"
			" multiple feature blob names and leveldb names seperated by ','."
			" The names cannot contain white space characters and the number of blobs"
			" and leveldbs must be equal.";
		return 1;
	}
	int arg_pos = num_required_args;

	arg_pos = num_required_args;
	if (argc > arg_pos && strcmp(argv[arg_pos], "GPU") == 0) {
		LOG(ERROR)<< "Using GPU";
		int device_id = 0;
		if (argc > arg_pos + 1) {
			device_id = atoi(argv[arg_pos + 1]);
			CHECK_GE(device_id, 0);
		}
		LOG(ERROR) << "Using Device_id=" << device_id;
		Caffe::SetDevice(device_id);
		Caffe::set_mode(Caffe::GPU);
	} else {
		LOG(ERROR) << "Using CPU";
		Caffe::set_mode(Caffe::CPU);
	}

	arg_pos = 0;  // the name of the executable
	string pretrained_binary_proto(argv[++arg_pos]);

	// Expected prototxt contains at least one data layer such as
	//  the layer data_layer_name and one feature blob such as the
	//  fc7 top blob to extract features.
	/*
	layers {
	name: "data_layer_name"
	type: DATA
	data_param {
	source: "/path/to/your/images/to/extract/feature/images_leveldb"
	mean_file: "/path/to/your/image_mean.binaryproto"
	batch_size: 128
	crop_size: 227
	mirror: false
	}
	top: "data_blob_name"
	top: "label_blob_name"
	}
	layers {
	name: "drop7"
	type: DROPOUT
	dropout_param {
	dropout_ratio: 0.5
	}
	bottom: "fc7"
	top: "fc7"
	}
	*/
	string feature_extraction_proto(argv[++arg_pos]);
	boost::shared_ptr<Net<Dtype> > feature_extraction_net(
		new Net<Dtype>(feature_extraction_proto, caffe::TEST));
	feature_extraction_net->CopyTrainedLayersFromBinaryProto(pretrained_binary_proto);

	string extract_feature_blob_names(argv[++arg_pos]);
	vector<string> blob_names;
	boost::split(blob_names, extract_feature_blob_names, boost::is_any_of(","));

	string save_feature_leveldb_names(argv[++arg_pos]);
	vector<string> leveldb_names;
	boost::split(leveldb_names, save_feature_leveldb_names,
		boost::is_any_of(","));
	CHECK_EQ(blob_names.size(), leveldb_names.size()) <<
		" the number of blob names and leveldb names must be equal";
	size_t num_features = blob_names.size();

	for (size_t i = 0; i < num_features; i++) {
		CHECK(feature_extraction_net->has_blob(blob_names[i]))
			<< "Unknown feature blob name " << blob_names[i]
		<< " in the network " << feature_extraction_proto;
	}

	vector<std::ofstream*> feature_dbs;
	for(int i=0;i<num_features;i++)
	{
		LOG(INFO) << "opening feature files:" << leveldb_names[i];
		//output to binary file
		std::ofstream* db = new std::ofstream(leveldb_names[i], ios::binary);
		feature_dbs.push_back(db);
	}

	//num_mini_batches * batch_size = num of test images
	int num_mini_batches = atoi(argv[++arg_pos]);

	LOG(INFO)<< "Extacting Features";
	
	vector<Blob<Dtype>*> input_vec;
	vector<int> image_indices(num_features, 0);
	for (int batch_index = 0; batch_index < num_mini_batches; ++batch_index) {

		feature_extraction_net->Forward(input_vec);
		for (int i = 0; i < num_features; ++i) {
			const boost::shared_ptr<Blob<Dtype> > feature_blob = feature_extraction_net
				->blob_by_name(blob_names[i]);
			int batch_size = feature_blob->num();
			int dim_features = feature_blob->count() / batch_size;
			Dtype* feature_blob_data;
			for (int n = 0; n < batch_size; ++n) {
				feature_blob_data = feature_blob->mutable_cpu_data() +
					feature_blob->offset(n);
				//square root
				if (FLAGS_sqrt){
					caffe_powx<Dtype>(dim_features, feature_blob_data,
						Dtype(0.5), feature_blob_data);
				}
				//l2 norm normalize
				if (FLAGS_l2_norm){
					L2Normalize(dim_features, feature_blob_data);
				}
				for (int d = 0; d < dim_features; ++d) {
					*feature_dbs[i] << feature_blob_data[d] << ' ';   
				}
//				*feature_dbs[i]<<'\n';
				++image_indices[i];
				if (image_indices[i] % 1000 == 0) {
					LOG(ERROR)<< "Extracted features of " << image_indices[i] <<
						" query images for feature blob " << blob_names[i];
				}
			}  // for (int n = 0; n < batch_size; ++n)
		}  // for (int i = 0; i < num_features; ++i)
	}  // for (int batch_index = 0; batch_index < num_mini_batches; ++batch_index)
	// write the last batch
	for (int i = 0; i < num_features; ++i) {
		LOG(ERROR)<< "Extracted features of " << image_indices[i] <<
			" query images for feature blob " << blob_names[i];
	}

	for(int i=0;i<num_features;i++)
	{
		(*feature_dbs[i]).close();
		delete feature_dbs[i];
	}
	LOG(ERROR)<< "Successfully extracted the features!";
	return 0;
}

