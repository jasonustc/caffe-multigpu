#include <cstring>
#include <cstdlib>
#include <vector>

#include <fstream>
#include <iostream>

#include "boost/algorithm/string.hpp"
#include "google/protobuf/text_format.h"
#include "leveldb/db.h"
#include "leveldb/write_batch.h"
#include "glog/logging.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/net.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"
#include "caffe/vision_layers.hpp"

using namespace caffe;  // NOLINT(build/namespaces)

DEFINE_int32(pool, 0, "pooling type(0:average[default], 1:sum, 2: max) of last spatially average of the class score map.");

template<typename Dtype>
int feature_extraction_pipeline(int argc, char** argv);

int main(int argc, char** argv) {
	::google::InitGoogleLogging(*argv);
	::google::SetStderrLogging(0);
	gflags::ParseCommandLineFlags(&argc, &argv, true);
	return feature_extraction_pipeline<float>(argc, argv);
	//  return feature_extraction_pipeline<double>(argc, argv);
}

template<typename Dtype>
int feature_extraction_pipeline(int argc, char** argv) {
	const int num_required_args = 6;
	if (argc < num_required_args) {
		LOG(ERROR)<<
			"This program takes in a trained network and an input data layer, and then"
			" extract features of the input data produced by the net.\n"
			"Usage: extract_features [FLAGS] pretrained_net_param"
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

	LOG(ERROR) << "feature pool type: " << FLAGS_pool;

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
	feature_extraction_net->CopyTrainedLayersFrom(pretrained_binary_proto);

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
		LOG(INFO) << blob_names[i];
		CHECK(feature_extraction_net->has_blob(blob_names[i]))
			<< "Unknown feature blob name " << blob_names[i]
		<< " in the network " << feature_extraction_proto;
	}

	

	vector<std::ofstream*> feature_dbs;
	for(int i=0;i<num_features;i++)
	{
		LOG(ERROR)<<"opening feature files:"<<leveldb_names[i];
		std::ofstream* db=new std::ofstream(leveldb_names[i]);
		feature_dbs.push_back(db);
	}

	//num_mini_batches * batch_size = num of test images
	int num_mini_batches = atoi(argv[++arg_pos]);

	LOG(ERROR)<< "Extacting Features";

	
	vector<Blob<float>*> input_vec;
	vector<int> image_indices(num_features, 0);
//	std::ofstream out_label_info("label_in_valid_lmdb.txt");
	/*
	 *NOTE: if size of the images are different, we must set batch_size to be 1.
	 *because different images will output different size of feature maps.
	 */
	for (int batch_index = 0; batch_index < num_mini_batches; ++batch_index) {
		feature_extraction_net->Forward(input_vec);
		for (int i = 0; i < num_features; ++i) {
//			const boost::shared_ptr<Blob<Dtype>> label_blob = feature_extraction_net
//				->blob_by_name("label");
//			int num_labels = label_blob->count();
//			for (int l = 0; l < num_labels; l++){
//				out_label_info << label_blob->cpu_data()[l] << "\n";
//			}
			const boost::shared_ptr<Blob<Dtype> > feature_blob = feature_extraction_net
				->blob_by_name(blob_names[i]);
			int batch_size = feature_blob->num();
//			LOG(INFO) << feature_blob->channels() << "x" << feature_blob->height() 
//				<< "x" << feature_blob->width();
			int dim_features = feature_blob->count() / batch_size;
			int channels = feature_blob->channels();
			int spatial_dim = feature_blob->count() / batch_size / channels;
			const Dtype* feature_blob_data;
			for (int n = 0; n < batch_size; ++n) {
				for (int c = 0; c < channels; ++c) {
					feature_blob_data = feature_blob->cpu_data() +
						feature_blob->offset(n, c);
					Dtype pool_feat = 0;
					switch (FLAGS_pool)
					{
					case 0:
						//average pooling
						for (int s = 0; s < spatial_dim; s++){
							pool_feat += feature_blob_data[s];
						}
						pool_feat /= spatial_dim;
						break;
					case 1:
						//sum pooling
						for (int s = 0; s < spatial_dim; s++){
							pool_feat += feature_blob_data[s];
						}
						break;
					case 2:
						//max pooling
						pool_feat = FLT_MIN;
						for (int s = 0; s < spatial_dim; s++){
							if (feature_blob_data[s] > pool_feat){
								pool_feat = feature_blob_data[s];
							}
						}
						break;
					default:
						LOG(FATAL) << "Unkown pool type of spatial features.";
						break;
					}
					*feature_dbs[i] << pool_feat << ' ';   //added by xu shen
				}
				*feature_dbs[i] << '\n';
				
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
//	out_label_info.close();
	LOG(ERROR)<< "Successfully extracted the features!";
	return 0;
}

