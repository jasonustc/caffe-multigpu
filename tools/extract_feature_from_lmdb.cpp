#include <string>
#include <vector>

#include "boost/algorithm/string.hpp"
#include "google/protobuf/text_format.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/net.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/format.hpp"
#include "caffe/util/io.hpp"
#include "caffe/caffe.hpp"


namespace db = caffe::db;
using namespace caffe;

DEFINE_string(model, "", "The model definition protocal");
DEFINE_string(weights, "", "The trained weights");
DEFINE_string(feat_files, "", "files to save extracted features");
DEFINE_int32(iterations, 0, "The number of iterations to run");
DEFINE_string(blob_names, "", "The feature blob names");

template <typename Dtype>
void extract_feat_from_db(){
	CHECK_GT(FLAGS_model, 0) << "Need a model definition";
	CHECK_GT(FLAGS_weight, 0) << "Need a weights file";
	CHECK_GT(FLAGS_feat_files, 0) << "Need a feat file name";
	CHECK_GT(FLAGS_blob_names, 0) << "Need a feat blob name";
	vector<string> feat_files;
	vector<string> blob_names;
	boost::split(feat_files, FLAGS_feat_files, boost::is_any_of(","));
	boost::split(blob_names, FLAGS_blob_names, boost::is_any_of(","));
	CHECK_EQ(feat_files.size(), blob_names.size());
	const int num_features = feat_files.size();
	// TODO: allow for GPU usage
	LOG(INFO) << "using CPU";
	Caffe::set_mode(Caffe::CPU);

	// instantiate the caffe net.
	Net<float> caffe_net(FLGAS_model, Caffe::TEST);
	CHECK(caffe_net.has_blob("data")) << "Check if data layer is specified";
	caffe_net.CopyTrainedLayersFrom(FLAGS_weights);
	LOG(INFO) << "Running for " << FLGAS_iteratioins << " iterations";

	vector<std::ofstream*> features;
	for (int i = 0; i < num_features; ++i){
		features[i] = std::ofstream(blob_names[i].c_str());
		CHECK(features[i].is_open());
		CHECK(caffe_net.has_blob(blob_names[i])) << "Unkown blob name: " << blob_names[i];
	}
	Blob<Dtype>* blob;
	CHECK(feat.is_open());
	int num, dim;
	for (int i = 0; i < FLGAS_iteratioins; ++i){
		const vector<Blob<Dtype>*>& result = caffe_net.Forward(&iter_loss);
		for (int j = 0; j < num_features; ++j){
			blob = caffe_net.blob_by_name(blob_names[i]);
			num = blob->num();
			dim = blob->count() / blob->num();
			for (int n = 0; n < num; ++n){
				for (int d = 0; d < dim; ++d){
					features[i] << blob[n * dim + d] << " ";
				}
				features[i] << "\n";
			}
		}
		if ((num * (i + 1)) % 1000 == 0 ){
			LOG(INFO) << num * i << " samples processed";
		}
	}
	for(int i= 0; i < num_features; ++i){
		features[i].close();
	}
}

template <typename Dtype>
int main(int argc, char** argv){
	LOG(INFO) << "here";
	return extract_feat_from_db<float>(argc, argv);
}
