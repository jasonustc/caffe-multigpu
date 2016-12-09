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
DEFINE_string(backend, "lmdb", "The backend{leveldb/lmdb} for storing the result");
DEFINE_int32(num_items, 0, "number of samples");

void convert_dataset_float(const string& feat_file, const string& label_file, const string& db_name) {
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
	CHECK_GT(FLAGS_num_items, 0) << "number of samples should be positive";
	datum.set_channels(FLAGS_channels);
	datum.set_height(FLAGS_height);
	datum.set_width(FLAGS_width);

	LOG(ERROR) << "Loading data...";
	const int dim = FLAGS_channels * FLAGS_height * FLAGS_width;
	int label = 0;
	int count = 0;
	float data;
	ifstream in_feat(feat_file.c_str());
	CHECK(in_feat.is_open());
	ifstream in_label(feat_file.c_str());
	CHECK(in_label.is_open());
	for (size_t f = 0; f < FLAGS_num_items; f++){
		in_label >> label;
		//save feat/label data to db
		datum.clear_float_data();
		for (int i = 0; i < dim; i++){
			in_feat >> data;
			datum.add_float_data(data);
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
	if (argc != 4) {
		gflags::SetUsageMessage("Convert feats and label to lmdb/leveldb\n"
			"format used for caffe.\n"
			"Usage: \n"
			"EXE [FLAGS] FEAT_FILE LABEL_FILE DB_NAME\n");
		gflags::ShowUsageWithFlagsRestrict(argv[0], "convert_mnist_data");
		return 1;
	}
	else {
		LOG(INFO) << "Channels: " << FLAGS_channels << ", height: " 
			<< FLAGS_height << ", width: " << FLAGS_width;
		convert_dataset_float(string(argv[1]), string(argv[2]), string(argv[3]));
	}
	return 0;
}
