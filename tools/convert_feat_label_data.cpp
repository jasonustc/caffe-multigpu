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
DEFINE_string(spliter, "", "The spliter to split each dim of feat in feat file");
DEFINE_bool(shuffle, false, "If we need to shuffle the order of feat files");
DEFINE_string(root_dir, "", "The root folder of feat files");

void parse_file_feat_v2(string& file_path, vector<float>& feat){
	feat.clear();
	ifstream in_feat(file_path.c_str());
	CHECK(in_feat.is_open()) << "can not open feat file " << file_path;
	float feat_i;
	while (in_feat >> feat_i){
		feat.push_back(feat_i);
	}
	in_feat.close();
}

void parse_file_feat(string& file_path, vector<float>& feat){
	feat.clear();
	vector<string> strs;
	ifstream in_feat(file_path.c_str());
	CHECK(in_feat.is_open()) << "can not open file " << file_path;
	//delete spaces in the beginning and ending of the sequence
	string line;
	string spliter = FLAGS_spliter.size() > 0 ? FLAGS_spliter : " ";
	while (getline(in_feat, line)){
		boost::trim(line);
		boost::split(strs, line, boost::is_any_of(spliter));
		float feat_i;
		for (vector<string>::iterator it = strs.begin();
			it != strs.end(); ++it){
			istringstream iss(*it);
			//to skip space input
			if ((*it).size() == 0){
				continue;
			}
			iss >> feat_i;
			feat.push_back(feat_i);
		}
	}
	in_feat.close();
}

void load_file_list(const string& file, vector<std::pair<string, int> >& file_list){
	ifstream in_list(file.c_str());
	CHECK(in_list.is_open()) << "Can not open list file: " << file.c_str();
	string file_path;
	int label;
	while (in_list >> file_path >> label){
		file_path = FLAGS_root_dir + file_path;
		file_list.push_back(std::make_pair(file_path, label));
	}
	if (FLAGS_shuffle){
		LOG(ERROR) << "shuffling data";
		shuffle(file_list.begin(), file_list.end());
	}
	in_list.close();
}

void convert_dataset_float (const string& list_file, const string& db_name) {
	vector<std::pair<string, int> > file_list;
	load_file_list(list_file, file_list);
	CHECK(file_list.size() > 0);

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
	for (size_t f = 0; f < file_list.size(); f++){
		parse_file_feat_v2(file_list[f].first, feats);
		//check feat dim 
		int feat_dim = FLAGS_channels * FLAGS_height * FLAGS_width;
		CHECK_EQ(feat_dim, feats.size()) << "Feat dim not match, required: "
			<< feat_dim << ", get: " << feats.size() 
			<< "\nFile: " << file_list[f].first;
		//save feat/label data to db
		datum.clear_float_data();
		for (int i = 0; i < feats.size(); i++){
			datum.add_float_data(feats[i]);
		}
		datum.set_label(file_list[f].second);
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
	if (argc < 3) {
		gflags::SetUsageMessage("Convert feats and scores by line to lmdb/leveldb\n"
			"format used for caffe.\n"
			"Usage: \n"
			"EXE [FLAGS] FEAT_FILE_INDEX(feat_file_index label) DB_NAME\n");
		gflags::ShowUsageWithFlagsRestrict(argv[0], "convert_feat_label_data");
		return 1;
	}
	else {
		LOG(INFO) << "Channels: " << FLAGS_channels << ", height: " 
			<< FLAGS_height << ", width: " << FLAGS_width;
		convert_dataset_float(string(argv[1]), string(argv[2]));
	}
	return 0;
}
