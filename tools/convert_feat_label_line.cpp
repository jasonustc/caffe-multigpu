#include <fstream>  // NOLINT(readability/streams)
#include <string>

#include "boost/scoped_ptr.hpp"
#include "glog/logging.h"
#include "google/protobuf/text_format.h"
#include "stdint.h"

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "boost/algorithm/string.hpp"
#include "caffe/util/rng.hpp"

using caffe::Datum;
using boost::scoped_ptr;
using namespace std;
using namespace caffe;

DEFINE_int32(channels, 0, "channels of the image");
DEFINE_int32(height, 0, "height of the image");
DEFINE_int32(width, 0, "width of the image");
DEFINE_int32(num_items, 0, "number of samples");
DEFINE_string(backend, "lmdb", "The backend{leveldb/lmdb} for storing the result");

void parse_line_feat(string line, vector<float>& feat, const int dim){
	if (line.empty()){
		LOG(FATAL) << "empty feat!";
	}
	feat.clear();
	std::stringstream ss;
	ss << line;
	float feat_i;
	while (ss >> feat_i){
		feat.push_back(feat_i);
	}
	CHECK_EQ(feat.size(), dim) << "loaded feat dim not as required";
}


void convert_dataset_float (const string& feat_file, const string& label_file, const string& db_name) {
	std::ifstream in_feat(feat_file.c_str());
	CHECK(in_feat.is_open());
	std::ifstream in_label(label_file.c_str());
	CHECK(in_label.is_open());

	//create db
	scoped_ptr<db::DB> db(db::GetDB(FLAGS_backend));
	db->Open(db_name, db::NEW);
	scoped_ptr<db::Transaction> txn(db->NewTransaction());

	// Data buffer
	const int kMaxKeyLength = 256;
	char key_cstr[kMaxKeyLength];
	Datum datum;

	CHECK(FLAGS_channels > 0 && FLAGS_height > 0 && FLAGS_width > 0)
		<< "channels, height and width should be positive; while it is set to be "
		<< FLAGS_channels << "," << FLAGS_height << "," << FLAGS_width;
	datum.set_channels(FLAGS_channels);
	datum.set_height(FLAGS_height);
	datum.set_width(FLAGS_width);

	LOG(ERROR) << "Loading data...";
	string line;
	vector<float> feats;
	int label = -1;
	int count = 0;
	CHECK_GT(FLAGS_num_items, 0);
	//check feat dim 
	int feat_dim = FLAGS_channels * FLAGS_height * FLAGS_width;
	// NOTE: need a match between num_items and #feats and #labels
	for (size_t f = 0; f < FLAGS_num_items; f++){
		string feat_str;
		getline(in_feat, feat_str);
		parse_line_feat(feat_str, feats, feat_dim);
		in_label >> label;
		//save feat/label data to db
		datum.clear_float_data();
		for (int i = 0; i < feats.size(); i++){
			datum.add_float_data(feats[i]);
		}
		datum.set_label(label);
		//sequential
		string out;
		datum.SerializeToString(&out);
		int len = sprintf_s(key_cstr, kMaxKeyLength, "%09d", count);
		//put into db
		txn->Put(std::string(key_cstr, len), out);

		if ((++count) % 1000 == 0){
			//commit db
			txn->Commit();
			txn.reset(db->NewTransaction());
			LOG(ERROR) << "Processed " << count << " feats";
		}
	}
	if (count % 1000 != 0){
		txn->Commit();
		LOG(ERROR) << "Processed " << count << " feats";
	}
}

int main(int argc, char** argv) {
	::google::InitGoogleLogging(argv[0]);
	::google::SetStderrLogging(0);
	//parse flags
	::gflags::ParseCommandLineFlags(&argc, &argv, true);
	if (argc < 4) {
		gflags::SetUsageMessage("Convert feats and labels by line to lmdb/leveldb\n"
			"format used for caffe.\n"
			"Usage: \n"
			"EXE [FLAGS] FEAT_FILE_INDEX LABEL_FILE_INDEX DB_NAME\n");
		gflags::ShowUsageWithFlagsRestrict(argv[0], "convert_feat_label_data");
		return 1;
	}
	else {
		LOG(INFO) << "Channels: " << FLAGS_channels << ", height: " 
			<< FLAGS_height << ", width: " << FLAGS_width;
		convert_dataset_float(string(argv[1]), string(argv[2]), string(argv[3]));
	}
	return 0;
}
